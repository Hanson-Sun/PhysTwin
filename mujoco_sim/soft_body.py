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

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.6 * element_size)
        gmsh.merge(str(path))
        # Repair the imported surface into geometry, then fill it. This is the
        # documented route from an STL/OBJ surface to a tetrahedral volume.
        gmsh.model.mesh.classifySurfaces(0.9, True, True, 180.0)
        gmsh.model.mesh.createGeometry()
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
    name: str = "object",
) -> tuple[str, dict[str, bytes]]:
    """Build a soft-body object asset from any watertight mesh.

    Returns the MJCF text for the object and the extra files it references, in
    the shape ``scene.build_scene`` expects. The object keeps the same
    ``<body name="object">`` wrapper as the rigid assets, so trajectories and
    bounds lookups address it by the usual name.
    """
    nodes, tets = tetrahedralize(mesh_path, element_size)
    # Centre on the origin and rest on the ground, matching what
    # place_objects_on_ground does for the rigid assets.
    nodes = nodes - np.array([nodes[:, 0].mean(), nodes[:, 1].mean(), nodes[:, 2].min()])

    mesh_name = f"{Path(name).name}.msh"
    xml = f"""<mujoco model="soft_{name}">

  <!-- Soft body tetrahedralized from {Path(mesh_path).name}. MuJoCo does not
       integrate flex elasticity under the scene's implicitfast integrator, so
       the soft asset selects the discrete integrator for the whole model. -->
  <option integrator="discrete" solver="Newton"/>

  <worldbody>
    <body name="{name}">
      <flexcomp name="{name}" type="gmsh" file="{mesh_name}" dof="{dof}"
                mass="{mass}" radius="{radius}" rgba="{rgba}">
        <elasticity young="{young}" poisson="{poisson}" damping="{damping}"/>
        <edge damping="{edge_damping}"/>
        <contact selfcollide="none" friction="{friction}" solref="0.02 1"/>
      </flexcomp>
    </body>
  </worldbody>

</mujoco>
"""
    return xml, {mesh_name: gmsh_bytes(nodes, tets)}


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
