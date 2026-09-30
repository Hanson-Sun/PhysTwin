"""Turn a shape-prior mesh into a watertight asset for the soft-body sim.

PhysTwin shape priors (``shape/matching/final_mesh.glb``) are surface
reconstructions: not watertight, dozens of disconnected components, and posed
in the capture frame where the object lies flat on the table. MuJoCo flex
bodies need a closed volume, so this closes the mesh by voxel-filling it.

When the input carries a UV-unwrapped texture (the TRELLIS .glb does), two
sidecars are written next to the output so the sim can port the appearance:

- ``<name>.png`` - the base-colour texture;
- ``<name>.uvsrc.npz`` - the source triangles and their per-corner UVs, in the
  same frame as the written mesh. ``soft_body.soft_object`` projects the
  tetrahedralized nodes onto these triangles to give the flex UVs; MuJoCo
  samples textures with v=0 at the image's top row, which is also glTF's
  convention, so the UVs are used as-is.

Rotations are extrinsic and compose in a fixed x, then y, then z order. The
sloth ships rolled onto its back and yawed so the head points at camera 0
(cam_front, -y):

Example:
    python -m mujoco_sim.prepare_mesh \\
        data/different_types/double_lift_sloth/shape/matching/final_mesh.glb \\
        mujoco_assets/object_sloth.stl \\
        --rotate-x-deg 180 --rotate-z-deg 90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

# ~4 voxels per claw-pad radius: still bridges the millimetre-scale gaps in a
# fragmented reconstruction, but rounds silhouette detail no coarser than
# ~6 mm (features below ~2 voxels), so ears and limbs stay readable.
DEFAULT_PITCH = 0.003


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file, concatenating every geometry in a scene (e.g. a GLB).

    Vertices are welded by position: GLB exporters split them across UV and
    normal seams, which would otherwise make a geometrically closed mesh read
    as open and force the voxel repair it does not need.
    """
    loaded = trimesh.load(path, force=None)
    if isinstance(loaded, trimesh.Scene):
        parts = list(loaded.geometry.values())
        if not parts:
            raise ValueError(f"{path} contains no geometry")
        loaded = trimesh.util.concatenate(parts)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f"{path} does not contain a triangle mesh")
    # Rebuild without visuals so merge_vertices welds purely by position
    # instead of keeping texture-seam vertices apart.
    welded = trimesh.Trimesh(
        vertices=loaded.vertices.copy(), faces=loaded.faces.copy(), process=False
    )
    welded.merge_vertices()
    welded.update_faces(welded.nondegenerate_faces())
    welded.remove_unreferenced_vertices()
    return welded


def uv_source(path: str | Path):
    """Return ``(triangles, corner_uvs, image)`` for a textured mesh, else None.

    ``triangles`` is ``[n, 3, 3]`` world-space corners and ``corner_uvs`` the
    matching ``[n, 3, 2]`` UVs.  None when the mesh has no UV-unwrapped base
    colour texture, i.e. when there is no appearance to port.  Geometries are
    read individually because concatenating a scene drops UVs.
    """
    loaded = trimesh.load(path, force=None)
    parts = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    if not parts:
        return None
    triangles, corners, image = [], [], None
    for part in parts:
        uv = getattr(getattr(part, "visual", None), "uv", None)
        if uv is None or len(uv) != len(part.vertices):
            return None  # some geometry has no UVs: do not port a partial texture
        triangles.append(part.vertices[part.faces])
        corners.append(np.asarray(uv)[part.faces])
        texture = getattr(getattr(part.visual, "material", None), "baseColorTexture", None)
        if texture is not None and image is None:
            image = texture
    if image is None:
        return None
    return np.concatenate(triangles), np.concatenate(corners), image


def watertight(
    mesh: trimesh.Trimesh, pitch: float = DEFAULT_PITCH, remesh: bool = False
) -> trimesh.Trimesh:
    """Close a fragmented surface into one watertight shell.

    Voxelising fills the holes between disconnected patches and bridges
    sub-voxel gaps; re-extracting the grid gives a closed, manifold surface
    that ``soft_body.tetrahedralize`` can turn into tetrahedra.

    An already-closed, single-shell input skips the round-trip and only gets
    its winding repaired: the voxel pipeline exists to close broken meshes,
    and on a clean one the round-trip would only add voxel artefacts. Pass
    ``remesh=True`` to force it anyway - useful to decimate a dense input,
    since gmsh pins every surface vertex as a tet node.
    """
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")
    if not remesh and mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1:
        closed = mesh.copy()
        closed.fix_normals()  # gmsh needs a consistent outward shell
        if closed.volume > 0.0:
            return closed
    voxels = mesh.voxelized(pitch=pitch).fill()
    closed = voxels.marching_cubes
    closed.apply_transform(voxels.transform)
    closed.merge_vertices()
    closed.update_faces(closed.nondegenerate_faces())
    if not closed.is_watertight:
        raise ValueError(f"voxel fill at pitch {pitch} did not close the mesh")
    return closed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mesh", type=Path, help="input mesh (e.g. a shape-prior .glb)")
    parser.add_argument("output", type=Path, help="asset to write (.stl/.obj/.ply)")
    parser.add_argument(
        "--pitch", type=float, default=DEFAULT_PITCH, help="voxel edge length in metres"
    )
    parser.add_argument(
        "--remesh",
        action="store_true",
        help="force the voxel round-trip even for an already-watertight input "
        "(also decimates a dense mesh, which gmsh would otherwise pin vertex "
        "by vertex)",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=None,
        help="scale the mesh so its height (z extent after the rotations) "
        "equals this many metres, e.g. 0.15",
    )
    parser.add_argument(
        "--max-dim",
        type=float,
        default=None,
        help="cap the scale so no bounding-box extent (after the rotations) "
        "exceeds this many metres - keeps wide objects inside the claw's "
        "0.20 m grip gap",
    )
    parser.add_argument(
        "--rotate-x-deg",
        type=float,
        default=0.0,
        help="roll about x before writing; 180 lays a face-down mesh on its back",
    )
    parser.add_argument(
        "--rotate-y-deg",
        type=float,
        default=0.0,
        help="rotate about y before writing; 90 stands the captured-thin axis "
        "along x, the claw's grip axis",
    )
    parser.add_argument(
        "--rotate-z-deg",
        type=float,
        default=0.0,
        help="yaw about z before writing (applied after the x and y rotations)",
    )
    args = parser.parse_args()

    transform = None
    for deg, axis in (
        (args.rotate_x_deg, (1, 0, 0)),
        (args.rotate_y_deg, (0, 1, 0)),
        (args.rotate_z_deg, (0, 0, 1)),
    ):
        if deg:
            step = trimesh.transformations.rotation_matrix(np.deg2rad(deg), axis)
            transform = step if transform is None else step @ transform

    source = uv_source(args.mesh)
    mesh = load_mesh(args.mesh)
    # Scale before voxelising so ``--pitch`` stays in written metres; the
    # posed extents decide the factor so height/max-dim account for rotations.
    scale = 1.0
    if args.target_height is not None or args.max_dim is not None:
        posed = mesh.vertices
        if transform is not None:
            posed = trimesh.transform_points(posed, transform)
        extents = posed.max(axis=0) - posed.min(axis=0)
        if args.target_height is not None:
            scale = args.target_height / extents[2]
        if args.max_dim is not None:
            scale = min(scale, args.max_dim / extents.max())
        mesh.apply_scale(scale)
        print(f"scaled \u00d7{scale:.4f}: posed extents {np.round(extents * scale, 4)} m")
    mesh = watertight(mesh, args.pitch, remesh=args.remesh)
    if transform is not None:
        mesh.apply_transform(transform)
    if source is not None and (transform is not None or scale != 1.0):
        # keep the UV source in the same frame as the written mesh
        port = np.eye(4) if transform is None else transform.copy()
        if scale != 1.0:
            port = port @ np.diag([scale, scale, scale, 1.0])
        source = (
            trimesh.transform_points(source[0].reshape(-1, 3), port).reshape(-1, 3, 3),
            source[1],
            source[2],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(
        f"wrote {args.output}: {len(mesh.vertices)} vertices, watertight, "
        f"extents {np.round(mesh.extents, 4)} m"
    )
    if source is not None:
        triangles, corners, image = source
        image.save(args.output.with_suffix(".png"))
        np.savez_compressed(args.output.with_suffix(".uvsrc.npz"), tri=triangles, uv=corners)
        print(
            f"wrote {args.output.with_suffix('.png').name} + "
            f"{args.output.with_suffix('.uvsrc.npz').name}: appearance ported "
            f"from {len(triangles)} textured triangles"
        )
    else:
        print(f"{args.mesh} has no UV texture: appearance stays a flat colour")


if __name__ == "__main__":
    main()
