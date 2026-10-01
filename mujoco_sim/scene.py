"""Composes the digital-twin scene from standalone MJCF files in assets/.

Each piece (world/cameras, claw, object) is a real, independently-loadable
.xml file -- open any of them directly in a MuJoCo viewer to inspect or
tweak it. This module just assembles them via MuJoCo's native <include>
mechanism and compiles the result; it does not build XML by hand.
"""

from pathlib import Path

import io
import tempfile

import mujoco
import numpy as np

try:
    from . import soft_body
except ImportError:  # Support running this file directly from the mujoco_sim directory.
    import soft_body

# MuJoCo distributes optional elasticity/sensor/actuator plugins alongside the
# Python package. Loading the bundled directory is harmless when the bindings
# already loaded it and makes plugin-backed MJCF work in direct script runs.
try:
    mujoco.mj_loadAllPluginLibraries(str(mujoco.PLUGINS_DIR))
except (AttributeError, OSError):
    # Older MuJoCo builds may not expose bundled plugin loading.
    pass

ASSETS_DIR = Path(__file__).parent / "../" / "mujoco_assets"

WORLD_FILE = "world.xml"
CLAW_MATERIALS_FILE = "claw_materials.xml"
CLAW_FILE = "claw.xml"
DEFAULT_OBJECT_FILE = "object_box.xml"


def _read(filename: str) -> str:
    return (ASSETS_DIR / filename).read_text()


# A mocap body has infinite mass and its velocity is *imposed* each step, not
# solved. Contact friction acts on relative velocity, so against a mocap claw the
# friction constraint cannot arrest the pad's motion: the pads slide straight up
# through the object and the grip carries no load (measured: 0% of the pad's
# motion transferred, versus 99% with a finite-mass claw). Driving the claw with
# position servos on a finite-mass body lets friction engage. It stays opt-in
# because the default mocap path is what the constraint-based grasp
# (``mujoco_sim.grasp``) and the existing datasets were built against.
#
# Slide damping and the servo force range are sized for the claw's own ~11.9 N
# weight: too little and it folds up under gravity instead of tracking.
_CLAW_SLIDE_AXES = ((0, 1.0, 0.0, 0.0), (1, 0.0, 1.0, 0.0), (2, 0.0, 0.0, 1.0))
_CLAW_SLIDE_DAMPING = 80.0
_CLAW_SERVO_GAIN = 6000.0
_CLAW_SERVO_FORCE = 400.0


def _make_interactor_dynamic(spec: mujoco.MjSpec) -> None:
    """Turn every mocap interactor in ``spec`` into a servo-driven finite-mass body.

    Uses MjSpec rather than editing the MJCF text: the claw body is looked up by
    name, so renaming it fails here instead of silently producing a model that
    still has a mocap claw.
    """
    for body in spec.bodies:
        if not body.mocap:
            continue
        name = body.name
        body.mocap = False
        for axis, x, y, z in _CLAW_SLIDE_AXES:
            joint = body.add_joint()
            joint.name = f"{name}_{'xyz'[axis]}"
            joint.type = mujoco.mjtJoint.mjJNT_SLIDE
            joint.axis = [x, y, z]
            # MjsJoint.damping wants a (3, 1) array even for a slide joint.
            joint.damping = np.full((3, 1), _CLAW_SLIDE_DAMPING)

            actuator = spec.add_actuator()
            actuator.name = f"{name}_{'xyz'[axis]}_motor"
            actuator.target = joint.name
            actuator.set_to_position(kp=_CLAW_SERVO_GAIN)
            # set_to_position sets the gain but leaves the transmission
            # undefined, which fails to compile.
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.ctrllimited = True
            actuator.ctrlrange = [-2.0, 2.0]
            actuator.forcelimited = True
            actuator.forcerange = [-_CLAW_SERVO_FORCE, _CLAW_SERVO_FORCE]


def build_scene(
    n_interactors: int = 1,
    object_file: str = DEFAULT_OBJECT_FILE,
    object_mesh: str | None = None,
    soft: dict | None = None,
):
    """Compose world + `n_interactors` claws + one object into a single model.

    `object_file` is a filename under assets/ (or an absolute path) to any
    standalone MJCF file containing an ``object`` body with a freejoint. The
    default is the validated rigid box; use a compatible volume asset when
    one is available.

    `object_mesh` is a mesh filename under assets/ (or an absolute path) to
    simulate as a soft body instead, in which case `object_file` is unused.
    The mesh is tetrahedralized on the fly; `soft` overrides the material and
    meshing defaults of `soft_body.soft_object`.

    Each interactor also gets an inactive `*_grasp_weld` equality, which
    `mujoco_sim.grasp` enables at grasp time to carry a gripped object.

    Returns (xml: str, assets: dict[str, bytes]), ready for
    ``compile_scene``.
    """
    if n_interactors < 0:
        raise ValueError("n_interactors cannot be negative")

    claw_template = _read(CLAW_FILE)
    extra_assets = {}
    if object_mesh is not None:
        mesh_path = Path(object_mesh)
        if not mesh_path.is_file():
            # A bare filename names an asset under mujoco_assets/.
            mesh_path = ASSETS_DIR / object_mesh
        object_name = "soft_object.xml"
        object_text, extra_assets = soft_body.soft_object(mesh_path, **(soft or {}))
    else:
        object_path = Path(object_file)
        object_name = object_path.name
        object_text = (
            object_path.read_text() if object_path.is_absolute() else _read(object_file)
        )

    assets = {
        WORLD_FILE: _read(WORLD_FILE).encode(),
        CLAW_MATERIALS_FILE: _read(CLAW_MATERIALS_FILE).encode(),
        object_name: object_text.encode(),
        **extra_assets,
    }

    includes = [
        f'<include file="{WORLD_FILE}"/>',
        f'<include file="{object_name}"/>',
    ]
    for i in range(n_interactors):
        name = f"interactor{i}"
        claw_file = f"claw_{name}.xml"
        claw_text = claw_template.replace("__NAME__", name)
        if i > 0:
            claw_text = claw_text.replace(
                f'<include file="{CLAW_MATERIALS_FILE}"/>', ""
            )
        assets[claw_file] = claw_text.encode()
        includes.append(f'<include file="{claw_file}"/>')

    # One inactive grasp weld per claw: `mujoco_sim.grasp` enables it at grasp
    # time to carry a rigid object with the claw (see GraspAttachment).
    welds = "".join(
        f'    <weld name="interactor{i}_grasp_weld" body1="interactor{i}"'
        ' body2="object" active="false"/>\n'
        for i in range(n_interactors)
    )
    equality = f"  <equality>\n{welds}  </equality>\n" if welds else ""
    xml = (
        '<mujoco model="digital_twin_scaffold">\n'
        + "\n".join(includes)
        + "\n"
        + equality
        + "</mujoco>\n"
    )
    return xml, assets


def load_model(
    n_interactors: int = 1,
    object_file: str = DEFAULT_OBJECT_FILE,
    object_mesh: str | None = None,
    soft: dict | None = None,
    interactor_dynamic: bool = False,
) -> mujoco.MjModel:
    """Compose and compile in one call."""
    xml, assets = build_scene(n_interactors, object_file, object_mesh, soft)
    return compile_scene(xml, assets, interactor_dynamic)


def compile_scene(
    xml: str, assets: dict[str, bytes], interactor_dynamic: bool = False
) -> mujoco.MjModel:
    """Compile ``(xml, assets)`` into a model.

    Two things MJCF cannot express are patched in through MjSpec, the only API
    that can edit a compiled-away flex and add joints to a compiled body:

    - ``soft_body.soft_object`` passes flex texture coordinates in the asset dict
      (see ``soft_body.TEXCOORDS_ASSET``). The material is already referenced by
      the object's MJCF; the patch also clears the flex's rgba so the texture is
      not tinted by the otherwise-standalone colour.
    - ``interactor_dynamic`` swaps each mocap claw for a servo-driven
      finite-mass body (see ``_make_interactor_dynamic``).

    With neither set -- the default rigid scene -- the XML compiles directly,
    so the common path skips MjSpec entirely.
    """
    assets = dict(assets)
    texcoords = assets.pop(soft_body.TEXCOORDS_ASSET, None)
    if texcoords is None and not interactor_dynamic:
        return mujoco.MjModel.from_xml_string(xml, assets)

    with tempfile.TemporaryDirectory() as mesh_dir:
        # MuJoCo reads <include> and <asset> files (textures) from the spec's
        # own dicts, but flexcomp's gmsh file through the separate VFS/disk
        # layer -- and MjSpec refuses both at once. So the generated mesh goes
        # to a real file for the duration of the compile, and the object's MJCF
        # is pointed at it. It only has to live until spec.compile() returns:
        # mjModel keeps a copy.
        files = dict(assets)
        for key, blob in assets.items():
            if key.endswith(".msh"):
                target = Path(mesh_dir) / key
                target.write_bytes(blob)
                for name, text in files.items():
                    if name.endswith(".xml"):
                        files[name] = text.replace(
                            f'file="{key}"'.encode(), f'file="{target}"'.encode()
                        )
        spec = mujoco.MjSpec.from_string(
            xml,
            include={k: v for k, v in files.items() if k.endswith(".xml")},
            assets={k: v for k, v in files.items() if not k.endswith(".xml")},
        )
        if interactor_dynamic:
            _make_interactor_dynamic(spec)
        if texcoords is not None:
            uv = np.load(io.BytesIO(texcoords))["uv"]
            flex = next((f for f in spec.flexes if f.name == "object"), None)
            if flex is None:
                raise ValueError(
                    f"texcoords supplied but no 'object' flex in {list(f.name for f in spec.flexes)}"
                )
            flex.texcoord = uv.ravel().tolist()
            flex.elemtexcoord = np.asarray(flex.elem, dtype=int).ravel().tolist()
            flex.rgba = [1.0, 1.0, 1.0, 1.0]  # texture carries the colour now
        return spec.compile()
