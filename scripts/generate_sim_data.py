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

from mujoco_sim.interactor import object_push_trajectory
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


def normalize(model: dict, manifest: Path) -> dict:
    object_file = model.get("object_file", DEFAULT_OBJECT_FILE)
    object_path = Path(object_file)
    if not object_path.is_absolute():
        for candidate in (manifest.parent / object_path, REPO_ROOT / object_path):
            if candidate.is_file():
                object_file = str(candidate.resolve())
                break

    case_name = model.get("case_name") or model.get("name")
    if not case_name:
        case_name = object_path.stem

    return {
        "case_name": str(case_name),
        "object_file": str(object_file),
        "n_interactors": integer(model, "n_interactors", 1),
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
        load_model(model["n_interactors"], model["object_file"]),
        width=model["width"],
        height=model["height"],
    )
    trajectory = object_push_trajectory(
        sim.model,
        sim.data,
        steps_per_segment=model["steps_per_segment"],
    )
    trajectories = {
        f"interactor{i}": trajectory for i in range(model["n_interactors"])
    }
    with tqdm(
        total=len(trajectory),
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
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    models = [normalize(model, manifest) for model in load_manifest(manifest)]
    output_dir = args.output_dir.resolve()
    for model in models:
        print(f"{model['case_name']}: {model['object_file']}")
        if not args.dry_run:
            print(f"  wrote {generate(model, output_dir, args.overwrite)}")


if __name__ == "__main__":
    main()
