"""Generate PhysTwin-compatible RGB-D data for every model in a JSON manifest.

Usage:
    python scripts/generate_sim_data.py models.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_sim.interactor import (
    object_grip_lift_trajectory,
    object_push_trajectory,
)
from mujoco_sim.phystwin_export import export_case
from mujoco_sim.scene import DEFAULT_OBJECT_FILE, load_model
from mujoco_sim.simulation import DigitalTwinSim

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "different_types"

def integer(row: dict, name: str, default: int) -> int:
    value = row.get(name, default)
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def positive_float(row: dict, name: str, default: float | None) -> float | None:
    value = row.get(name, default)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def nonnegative_float(row: dict, name: str, default: float) -> float:
    value = row.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return value


def grasp_mode(row: dict) -> str:
    """How a closed grip is carried: a grasp constraint or pad friction alone.

    ``physical`` is the default: the claw is a finite-mass servo-driven body, so
    pad friction can actually carry the load (see ``interactor_dynamic``), which
    is the honest behaviour to train on. ``constraint`` stays available
    explicitly for the weld-based carry.
    """
    mode = row.get("grasp_mode", "physical")
    if mode not in ("constraint", "physical"):
        raise ValueError("grasp_mode must be 'constraint' or 'physical'")
    return str(mode)


def boolean(row: dict, name: str, default: bool) -> bool:
    value = row.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be true or false")
    return value


def interactor_dynamic(row: dict) -> bool:
    """Whether the interactor is a servo-driven finite-mass body.

    A mocap claw has infinite mass and an imposed velocity, so contact friction
    can never hold it: the pads slide up through the object and the grip carries
    nothing (measured 0% of the pad's motion transferred, vs 99% with a
    finite-mass claw). Friction is therefore the whole mechanism in
    ``grasp_mode: "physical"``, so that mode turns this on by default. It can
    still be set explicitly to study the kinematic claw.
    """
    default = grasp_mode(row) == "physical"
    return boolean(row, "interactor_dynamic", default)


def load_manifest(path: Path) -> list[dict]:
    if path.suffix.lower() != ".json":
        raise ValueError("manifest must be a .json file")
    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    models = document.get("models") if isinstance(document, dict) else document
    if not isinstance(models, list) or not models:
        raise ValueError("JSON manifest must contain a non-empty 'models' list")
    if not all(isinstance(model, dict) for model in models):
        raise ValueError("every model entry must be an object")
    return models


def resolve_asset(value: str, manifest: Path) -> str:
    """Resolve an asset next to the manifest or the repo root.

    Anything that is not found is left alone, so bare names still resolve
    against the `mujoco_assets/` directory when the scene is composed.
    """
    path = Path(value)
    if not path.is_absolute():
        for candidate in (manifest.parent / path, REPO_ROOT / path):
            if candidate.is_file():
                return str(candidate.resolve())
    return str(value)


def normalize(model: dict, manifest: Path) -> dict:
    object_file = resolve_asset(model.get("object_file", DEFAULT_OBJECT_FILE), manifest)
    object_mesh = model.get("object_mesh")
    if object_mesh is not None:
        object_mesh = resolve_asset(str(object_mesh), manifest)

    soft = model.get("soft") or {}
    if not isinstance(soft, dict):
        raise ValueError("'soft' must be an object of soft_body.soft_object arguments")

    case_name = model.get("case_name") or model.get("name")
    if not case_name:
        stem = Path(object_mesh) if object_mesh else Path(object_file)
        case_name = stem.stem

    return {
        "case_name": str(case_name),
        "object_file": str(object_file),
        "object_mesh": object_mesh,
        "soft": soft,
        "n_interactors": integer(model, "n_interactors", 1),
        "trajectory": str(model.get("trajectory", "push")),
        "gripper_open_angle": positive_float(model, "gripper_open_angle", 0.70),
        # Hinge target at full close. 0.30 presses slightly into the object;
        # 0.0 commands the fingers fully together, so the pinch is limited only
        # by the force clamp (grip_force) and the contact forces.
        "gripper_closed_angle": nonnegative_float(model, "gripper_closed_angle", 0.30),
        # 'physical' (default): the lift is carried by pad friction alone, on a
        # finite-mass servo-driven claw. 'constraint': the object is carried by
        # a grasp constraint once the grip closes (see mujoco_sim/grasp.py).
        "grasp_mode": grasp_mode(model),
        # Finite-mass servo-driven claw. Required for a physical (friction-only)
        # grip to hold anything; see interactor_dynamic.
        "interactor_dynamic": interactor_dynamic(model),
        # Per-finger squeeze limit in hinge torque (N*m).
        "grip_force": positive_float(model, "grip_force", 2.0),
        "max_controller_speed": positive_float(
            model, "max_controller_speed", None
        ),
        "controller_free_speed": positive_float(
            model, "controller_free_speed", 1.0
        ),
        "controller_slowdown_epsilon": nonnegative_float(
            model, "controller_slowdown_epsilon", 0.0
        ),
        "width": integer(model, "width", 848),
        "height": integer(model, "height", 480),
        "steps_per_segment": integer(model, "steps_per_segment", 300),
        "capture_every": integer(model, "capture_every", 8),
        "substeps": integer(model, "substeps", 4),
        "fps": integer(model, "fps", 30),
    }


def generate(model: dict, output_dir: Path, overwrite: bool) -> Path:
    case_dir = output_dir / model["case_name"]
    if case_dir.exists() and any(case_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"case already exists: {case_dir}; use --overwrite")

    sim = DigitalTwinSim(
        load_model(
            model["n_interactors"],
            model["object_file"],
            model["object_mesh"],
            model["soft"],
            interactor_dynamic=model["interactor_dynamic"],
        ),
        width=model["width"],
        height=model["height"],
        gripper_open_angle=model["gripper_open_angle"],
        gripper_closed_angle=model["gripper_closed_angle"],
        grip_force=model["grip_force"],
        interactor_dynamic=model["interactor_dynamic"],
    )
    # Recorded in the exported metadata so a dataset states how it was held.
    sim.grasp_mode = model["grasp_mode"]
    gripper_opening = None
    if model["trajectory"] == "grip_lift":
        if model["n_interactors"] != 1:
            raise ValueError("grip_lift requires exactly one interactor")
        trajectory, closing = object_grip_lift_trajectory(
            sim.model,
            sim.data,
            steps_per_segment=model["steps_per_segment"],
        )
        trajectories = {"interactor0": trajectory}
        gripper_opening = {"interactor0": closing}
    elif model["trajectory"] == "push":
        trajectory = object_push_trajectory(
            sim.model,
            sim.data,
            steps_per_segment=model["steps_per_segment"],
        )
        trajectories = {
            f"interactor{i}": trajectory for i in range(model["n_interactors"])
        }
    else:
        raise ValueError(
            f"unsupported trajectory '{model['trajectory']}' for {model['case_name']}"
        )

    # Build trajectories from the original scene pose; floor placement must not
    # change any controller waypoint. It only affects the initial object state.
    sim.place_objects_on_ground()

    # The attach engages when the grip closes (see DigitalTwinSim.rollout).
    grasped_body = (
        "object"
        if model["trajectory"] == "grip_lift" and model["grasp_mode"] == "constraint"
        else None
    )

    with tqdm(
        total=None if model["max_controller_speed"] is not None else trajectory_length,
        desc=model["case_name"],
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        frames = sim.rollout(
            trajectories,
            object_bodies=["object"],
            capture_every=model["capture_every"],
            substeps=model["substeps"],
            progress=progress.update,
            gripper_opening=gripper_opening,
            grasped_body=grasped_body,
            grasp_offset=(0.0, 0.0, 0.0),
            max_controller_speed=model["max_controller_speed"],
            controller_free_speed=model["controller_free_speed"],
            controller_slowdown_epsilon=model["controller_slowdown_epsilon"],
        )
    if len(frames) < 2:
        raise ValueError(
            f"{model['case_name']} produced {len(frames)} frame; "
            "reduce capture_every or increase steps_per_segment"
        )
    export_case(sim, frames, case_dir, fps=model["fps"])
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON model manifest")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Only generate the named case(s); repeatable. Defaults to every model in the manifest.",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    models = [normalize(model, manifest) for model in load_manifest(manifest)]
    if args.cases:
        known = {model["case_name"] for model in models}
        missing = [case for case in args.cases if case not in known]
        if missing:
            raise ValueError(
                f"case(s) not in manifest {manifest.name}: {', '.join(missing)}"
            )
        models = [model for model in models if model["case_name"] in args.cases]
    output_dir = args.output_dir.resolve()
    for model in models:
        asset = model["object_mesh"] or model["object_file"]
        print(
            f"{model['case_name']}: {asset} "
            f"({model['trajectory']}, interactors={model['n_interactors']})"
        )
        if not args.dry_run:
            print(f"  wrote {generate(model, output_dir, args.overwrite)}")


if __name__ == "__main__":
    main()