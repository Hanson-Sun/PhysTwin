"""Composes the digital-twin scene from standalone MJCF files in assets/.

Each piece (world/cameras, claw, object) is a real, independently-loadable
.xml file -- open any of them directly in a MuJoCo viewer to inspect or
tweak it. This module just assembles them via MuJoCo's native <include>
mechanism and compiles the result; it does not build XML by hand.
"""

from pathlib import Path

import mujoco

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


def build_scene(n_interactors: int = 1, object_file: str = DEFAULT_OBJECT_FILE):
    """Compose world + `n_interactors` claws + one object into a single model.

    `object_file` is a filename under assets/ (or an absolute path) to any
    standalone MJCF file containing an ``object`` body with a freejoint. The
    default is the validated rigid box; use a compatible volume asset when
    one is available.

    Returns (xml: str, assets: dict[str, bytes]), ready for
    mujoco.MjModel.from_xml_string(xml, assets).
    """
    if n_interactors < 0:
        raise ValueError("n_interactors cannot be negative")

    claw_template = _read(CLAW_FILE)
    object_path = Path(object_file)
    object_name = object_path.name
    object_text = (
        object_path.read_text() if object_path.is_absolute() else _read(object_file)
    )

    assets = {
        WORLD_FILE: _read(WORLD_FILE).encode(),
        CLAW_MATERIALS_FILE: _read(CLAW_MATERIALS_FILE).encode(),
        object_name: object_text.encode(),
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

    xml = (
        '<mujoco model="digital_twin_scaffold">\n'
        + "\n".join(includes)
        + "\n</mujoco>\n"
    )
    return xml, assets


def load_model(n_interactors: int = 1, object_file: str = DEFAULT_OBJECT_FILE) -> mujoco.MjModel:
    """Convenience: compose and compile in one call."""
    xml, assets = build_scene(n_interactors, object_file)
    return mujoco.MjModel.from_xml_string(xml, assets)
