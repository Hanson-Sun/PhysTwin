"""Turn an arbitrary watertight mesh into a MuJoCo soft-body object asset.

MuJoCo loads volumetric soft bodies only from GMSH tetrahedral meshes
(``flexcomp type="gmsh"``), so a surface mesh is tetrahedralized with gmsh and
serialised here. The GMSH writer is hand-rolled because gmsh's own writer splits
the nodes across several entity blocks, which MuJoCo rejects, and because it
emits surface elements alongside the tetrahedra.

Two MuJoCo requirements are easy to miss and are handled here:

- the object asset sets ``integrator="discrete"``: MuJoCo refuses to integrate
  flex elasticity under the scene's ``implicitfast`` integrator, so a soft body
  would otherwise silently do nothing;
- the mesh is recentred and rested on the ground, so the object needs no runtime
  placement (which ``DigitalTwinSim.place_objects_on_ground`` cannot do for a
  flex, since a flex has neither geometry nor a freejoint).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

# Surface mesh formats gmsh reads directly and MuJoCo mesh assets also accept.
MESH_EXTENSIONS = (".obj", ".stl", ".ply")

# The stiffness-to-mass ratio decides whether MuJoCo's flex preconditioner
# stays well conditioned (it warns above roughly young=1e5 at mass=0.5).
DEFAULT_ELEMENT_SIZE = 0.02
DEFAULT_MASS = 0.3
DEFAULT_RADIUS = 0.005
# Soft enough to flatten visibly on the floor (~4% of its width) and to squash
# in the claw's grip, without the stiffness-to-mass ratio upsetting MuJoCo's
# flex preconditioner.
DEFAULT_YOUNG = 1.0e4
DEFAULT_POISSON = 0.2
DEFAULT_DAMPING = 0.05
DEFAULT_EDGE_DAMPING = 1.0
DEFAULT_FRICTION = "0.4 0.005 0.0001"
# Deliberately not the rigid box's red: downstream masks are prompted on the
# object's colour, and the claw's orange contact spheres would be picked up
# instead of the object. Blue is distinct from every other body in the scene.
DEFAULT_RGBA = "0.10 0.30 0.90 1"
# MuJoCo integrates the "full" parametrization at ~30 ms per step for a 50 mm
# ball, which is far too slow to render a whole dataset. "trilinear" maps the
# elasticity onto a coarse grid for ~0.04 ms per step, at the cost of deforming
# roughly 2.5x less deeply. Note that edge damping only affects "full".
DEFAULT_DOF = "trilinear"

# Appearance port.  ``prepare_mesh`` writes ``<stem>.png`` (base-colour
# texture) and ``<stem>.uvsrc.npz`` (source triangles + corner UVs) next to a
# textured mesh; when both are present, ``soft_object`` gives the tetrahedral
# nodes UVs by projecting them onto those triangles.  MJCF cannot carry
# texcoords on a ``flexcomp``, so they travel in the returned asset dict under
# ``TEXCOORDS_ASSET`` and ``scene.load_model`` attaches them at compile time.
UV_SOURCE_SUFFIX = ".uvsrc.npz"
TEXTURE_SUFFIX = ".png"
TEXCOORDS_ASSET = "flex_texcoord.npz"
TEXTURE_MATERIAL = "object_texture"
TEXTURE_MAP = "object_texture_map"


class _SurfaceRepairError(RuntimeError):
    """gmsh could not re-parametrise an already closed surface mesh."""


def _gmsh_tets(path: Path, element_size: float, repair: bool):
    """Run gmsh over ``path`` and return raw ``(tags, coords, elements)``.

    With ``repair`` the imported facets are first turned back into smooth
    geometry, which is the documented route from an STL/OBJ triangle soup to a
    tetrahedral volume. A surface that is already a closed manifold (marching-
    cubes output from a voxel repair) has no seam for gmsh to snap together and
    fails there, so the caller retries with ``repair=False`` and lets the
    facets themselves bound the volume.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.6 * element_size)
        gmsh.merge(str(path))
        if repair:
            try:
                gmsh.model.mesh.classifySurfaces(0.9, True, True, 180.0)
                gmsh.model.mesh.createGeometry()
            except Exception as error:
                raise _SurfaceRepairError(str(error)) from error
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise ValueError(f"{path} contains no surface mesh")
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        raw_tags, coords, _ = gmsh.model.mesh.getNodes()
        _, elements = gmsh.model.mesh.getElementsByType(4)
    finally:
        gmsh.finalize()
    return raw_tags, coords, elements


def tetrahedralize(
    mesh_path: str | Path, element_size: float = DEFAULT_ELEMENT_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nodes [n,3], tets [m,4])`` for a watertight surface mesh.

    ``element_size`` is the target edge length of the tetrahedra. It trades mesh
    quality and simulation cost against how finely the soft body deforms.
    """
    if element_size <= 0.0:
        raise ValueError("element_size must be positive")
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh file not found: {path}")
    if path.suffix.lower() not in MESH_EXTENSIONS:
        raise ValueError(
            f"unsupported mesh format '{path.suffix}'; expected one of {MESH_EXTENSIONS}"
        )

    try:
        raw_tags, coords, elements = _gmsh_tets(path, element_size, repair=True)
    except _SurfaceRepairError:
        # The closed surface cannot be re-parametrised; its facets already
        # bound the volume, so mesh them directly.
        raw_tags, coords, elements = _gmsh_tets(path, element_size, repair=False)

    tags = np.asarray(raw_tags, dtype=int)
    nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
    tets = np.asarray(elements, dtype=int).reshape(-1, 4)
    if not len(tets):
        raise ValueError(
            f"{path} did not tetrahedralize: the mesh is probably not watertight"
        )

    # MuJoCo needs node tags running 1..n in order, and reads element node ids
    # as tags, so sort the nodes into tag order and rebase the tetrahedra.
    order = np.argsort(tags)
    nodes = nodes[order]
    if not np.array_equal(tags[order], np.arange(1, len(nodes) + 1)):
        raise ValueError(f"{path} produced non-sequential node tags")
    return nodes, tets - 1


def gmsh_bytes(nodes: np.ndarray, tets: np.ndarray) -> bytes:
    """Serialise a tetrahedral mesh as ASCII GMSH 4.1 with one node block."""
    lines = [
        "$MeshFormat",
        "4.1 0 8",
        "$EndMeshFormat",
        "$Nodes",
        f"1 {len(nodes)} 1 {len(nodes)}",  # one block, tags 1..n
        f"3 1 0 {len(nodes)}",  # 3D, non-parametric
    ]
    lines += [str(index + 1) for index in range(len(nodes))]
    lines += [f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in nodes]
    lines += ["$EndNodes", "$Elements", f"1 {len(tets)} 1 {len(tets)}", f"3 1 4 {len(tets)}"]
    lines += [
        f"{index + 1} " + " ".join(str(int(node) + 1) for node in tet)
        for index, tet in enumerate(tets)
    ]
    lines += ["$EndElements"]
    return ("\n".join(lines) + "\n").encode()


def _project_uvs(
    points: np.ndarray, tris: np.ndarray, corner_uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``points`` onto ``tris`` and return ``(uv, distance)``.

    ``points`` is ``[n, 1, 3]`` so it broadcasts over ``tris``'s ``[n, k, ...]``
    candidate axis. Barycentrics are clipped to the triangle so a distant
    candidate extrapolates no further than its own corners.
    """
    a, b, c = tris[..., 0, :], tris[..., 1, :], tris[..., 2, :]
    ab, ac, ap = b - a, c - a, points - a

    def dot(x, y):
        return np.einsum("...i,...i->...", x, y)

    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    d20, d21 = dot(ap, ab), dot(ap, ac)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-20, 1.0, denom)
    vb = (d11 * d20 - d01 * d21) / denom
    vc = (d00 * d21 - d01 * d20) / denom
    weights = np.clip(np.stack([1.0 - vb - vc, vb, vc], axis=-1), 0.0, 1.0)
    weights /= weights.sum(axis=-1, keepdims=True) + 1e-12
    uv = np.einsum("...i,...ij->...j", weights, corner_uv)
    closest = np.einsum("...i,...ij->...j", weights, tris)
    return uv, np.linalg.norm(points - closest, axis=-1)


def _texcoords(uv_source: Path, nodes: np.ndarray) -> np.ndarray:
    """Give ``nodes`` UVs by projecting them onto the UV-source triangles.

    The voxel repair replaces the source surface with marching-cubes output,
    so tetrahedral nodes carry no UVs; each node is projected onto nearby
    source triangles and takes the barycentric blend of their corner UVs.

    Picking just the triangle with the nearest centroid mis-assigns ~7% of
    nodes: the priors' UV charts interleave in 3D, and a large triangle's
    centroid can be far from the node. Keeping the k nearest triangles as
    candidates and taking whichever projects closest cuts that to ~0.4%,
    confined to spots where the voxel fill fused separate layers and no
    projection distance is decisive. Chart-continuity smoothing is
    deliberately absent: it re-charted correct nodes onto wrong atlas charts
    far more often than it resolved those ambiguous ones.
    """
    from scipy.spatial import cKDTree

    data = np.load(uv_source)
    tri = np.asarray(data["tri"], dtype=float)
    corner_uv = np.asarray(data["uv"], dtype=float)

    # Candidates: each node projected onto its k nearest source triangles.
    k = min(24, len(tri))
    _, candidates = cKDTree(tri.mean(axis=1)).query(nodes, k=k)
    candidates = np.asarray(candidates).reshape(len(nodes), -1)
    candidate_uv, candidate_dist = _project_uvs(
        nodes[:, None, :], tri[candidates], corner_uv[candidates]
    )
    uv = candidate_uv[np.arange(len(nodes)), np.argmin(candidate_dist, axis=1)]
    return np.clip(uv, 0.0, 1.0)


def soft_object(
    mesh_path: str | Path,
    *,
    element_size: float = DEFAULT_ELEMENT_SIZE,
    mass: float = DEFAULT_MASS,
    radius: float = DEFAULT_RADIUS,
    young: float = DEFAULT_YOUNG,
    poisson: float = DEFAULT_POISSON,
    damping: float = DEFAULT_DAMPING,
    edge_damping: float = DEFAULT_EDGE_DAMPING,
    friction: str = DEFAULT_FRICTION,
    rgba: str = DEFAULT_RGBA,
    dof: str = DEFAULT_DOF,
    cellcount: str | None = None,
    name: str = "object",
) -> tuple[str, dict[str, bytes]]:
    """Build a soft-body object asset from any watertight mesh.

    Returns the MJCF text for the object and the extra files it references, in
    the shape ``scene.build_scene`` expects. The object keeps the same
    ``<body name="object">`` wrapper as the rigid assets, so trajectories and
    bounds lookups address it by the usual name.

    ``cellcount`` (e.g. ``"4 6 1"``) is the reduced model's grid resolution.
    Left unset, MuJoCo covers the whole flex with a single 8-node cell, so it
    can only deform as one global affine warp: thin limbs cannot bend, and the
    shape bulges where the floor or the claw pushes it. Splitting the bounding
    box into cells lets neighbouring regions deform independently.

    When ``prepare_mesh`` left ``<stem>.png`` + ``<stem>.uvsrc.npz`` beside the
    mesh, the object also gets the source's texture: the flexcomp references a
    material and the returned assets carry the texture plus texcoords for
    ``scene.load_model`` to attach. In that case ``rgba`` only matters to
    callers that compile the raw MJCF, because the patch that wires the
    texture also sets the flex to white so the material is not tinted.
    """
    nodes, tets = tetrahedralize(mesh_path, element_size)
    texture_path = Path(mesh_path).with_suffix(TEXTURE_SUFFIX)
    uv_source_path = Path(mesh_path).with_suffix(UV_SOURCE_SUFFIX)
    # Project before recentring: the UV source is in the mesh's own frame.
    texcoords = (
        _texcoords(uv_source_path, nodes)
        if texture_path.is_file() and uv_source_path.is_file()
        else None
    )
    # Centre on the origin and rest on the ground, matching what
    # place_objects_on_ground does for the rigid assets. Vertices collide as
    # spheres of `radius`, so the mesh sits that far above the floor: grounding
    # the vertices themselves would start the body inside the floor and make it
    # balloon upward as the solver pushes it out.
    nodes = nodes - np.array(
        [nodes[:, 0].mean(), nodes[:, 1].mean(), nodes[:, 2].min() - radius]
    )

    mesh_name = f"{Path(name).name}.msh"
    cellcount_attr = f' cellcount="{cellcount}"' if cellcount else ""
    if texcoords is not None:
        material_attr = f' material="{TEXTURE_MATERIAL}"'
        appearance = f"""
  <asset>
    <texture name="{TEXTURE_MAP}" type="2d" file="{texture_path.name}"/>
    <material name="{TEXTURE_MATERIAL}" texture="{TEXTURE_MAP}"/>
  </asset>"""
    else:
        material_attr = ""
        appearance = ""
    xml = f"""<mujoco model="soft_{name}">{appearance}

  <!-- Soft body tetrahedralized from {Path(mesh_path).name}. MuJoCo does not
       integrate flex elasticity under the scene's implicitfast integrator, so
       the soft asset selects the discrete integrator for the whole model. -->
  <option integrator="discrete" solver="Newton"/>

  <worldbody>
    <body name="{name}">
      <flexcomp name="{name}" type="gmsh" file="{mesh_name}" dof="{dof}"{material_attr}
                mass="{mass}" radius="{radius}" rgba="{rgba}"{cellcount_attr}>
        <elasticity young="{young}" poisson="{poisson}" damping="{damping}"/>
        <edge damping="{edge_damping}"/>
        <contact selfcollide="none" friction="{friction}" solref="0.02 1"/>
      </flexcomp>
    </body>
  </worldbody>

</mujoco>
"""
    assets = {mesh_name: gmsh_bytes(nodes, tets)}
    if texcoords is not None:
        buffer = io.BytesIO()
        np.savez(buffer, uv=texcoords)
        assets[texture_path.name] = texture_path.read_bytes()
        assets[TEXCOORDS_ASSET] = buffer.getvalue()
    return xml, assets


def main() -> None:
    """Write a standalone soft-body asset next to a mesh, for inspection."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="watertight .obj/.stl/.ply mesh")
    parser.add_argument("object_file", type=Path, help="MJCF file to write")
    parser.add_argument("--element-size", type=float, default=DEFAULT_ELEMENT_SIZE)
    args = parser.parse_args()

    xml, assets = soft_object(args.mesh, element_size=args.element_size)
    args.object_file.parent.mkdir(parents=True, exist_ok=True)
    for name, data in assets.items():
        (args.object_file.parent / name).write_bytes(data)
    args.object_file.write_text(xml)
    print(f"wrote {args.object_file} and {', '.join(assets)}")


if __name__ == "__main__":
    main()
