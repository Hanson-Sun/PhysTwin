"""Warm-start Warp training over growing prefixes of one tracked PhysTwin case.

Configuration is JSON; see asid/README.md. All prefixes come from the same
final_data.pkl, preserving point identities and initial topology. This is not
yet an online multi-chunk RGB-D tracker or the complete active-learning loop.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


TIMESERIES = (
    "object_points", "object_colors", "object_visibilities", "object_motions_valid",
    "controller_points", "controller_points_dense",
)


def prepare_data(source: Path, destination: Path, frames: int, points: int | None) -> dict:
    """Write a tracked prefix with stable, deterministic object point identities."""
    with source.open("rb") as file:
        data = pickle.load(file)  # trusted project-generated data only
    object_points = np.asarray(data["object_points"])
    if object_points.ndim != 3 or object_points.shape[-1] != 3 or not 2 <= frames <= len(object_points):
        raise ValueError("requested frames do not match tracked object points")
    count = object_points.shape[1]
    if points is not None and not 1 <= points <= count:
        raise ValueError("object point count is outside the tracked dataset")
    selected = np.arange(count) if points is None else np.linspace(0, count - 1, points, dtype=int)
    for name in TIMESERIES:
        if name not in data:
            raise ValueError(f"tracked data missing {name}")
        value = np.asarray(data[name])
        if len(value) < frames:
            raise ValueError(f"{name} has fewer frames than requested")
        data[name] = value[:frames, selected].copy() if name.startswith("object_") else value[:frames].copy()
    for name in ("surface_points", "interior_points"):
        if name not in data:
            raise ValueError(f"tracked data missing {name}")
        if points is not None:
            data[name] = np.empty((0, 3), dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return {"frames": frames, "tracked_points": len(selected), "indices": selected.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    case = str(config["case_name"])
    base = Path(config["base_path"])
    output = Path(config["output_dir"])
    prefixes = config["frame_prefixes"]
    iterations = int(config.get("iterations_per_round", 1))
    points = config.get("tracked_points")
    if (not prefixes or any(not isinstance(v, int) or v < 2 for v in prefixes)
        or any(b <= a for a, b in zip(prefixes, prefixes[1:])) or iterations < 1):
        raise ValueError("frame_prefixes must increase from >=2 and iterations must be positive")
    if points is not None:
        points = int(points)
    case_dir = base / case
    source = case_dir / "final_data.pkl"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    previous = None
    results = []
    for round_index, frames in enumerate(prefixes, 1):
        round_dir = output / f"round_{round_index:03d}"
        round_dir.mkdir()
        data_info = prepare_data(source, round_dir / "final_data.pkl", frames, points)
        command = [
            sys.executable, "train_warp.py", "--base_path", str(base), "--case_name", case,
            "--train_frame", str(frames), "--iterations", str(iterations), "--no_visualize",
            "--data_path", str(round_dir / "final_data.pkl"), "--output_dir", str(round_dir),
        ]
        if previous is not None:
            command.extend(["--resume", str(previous)])
        with (round_dir / "training.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        checkpoints = list((round_dir / "train").glob("iter_*.pth"))
        if not checkpoints:
            raise RuntimeError(f"no checkpoint produced in {round_dir}")
        prior_checkpoint = previous
        previous = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        import torch
        checkpoint = torch.load(previous, map_location="cpu", weights_only=False)
        arrays = {
            name: checkpoint[name].detach().cpu().numpy()
            for name in ("spring_Y", "dashpot_damping", "drag_damping", "masses")
            if name in checkpoint
        }
        np.savez_compressed(round_dir / "parameters.npz", **arrays)
        adam_steps = [float(value["step"]) for value in checkpoint["optimizer_state_dict"]["state"].values() if "step" in value]
        result = {
            "round": round_index, "frames": frames, "epoch": int(checkpoint["epoch"]),
            "loss": float(checkpoint["training_loss"]),
            "checkpoint": str(previous), "warm_start_checkpoint": str(prior_checkpoint) if prior_checkpoint else None,
            "tracked_points": data_info["tracked_points"], "adam_steps": adam_steps,
        }
        if prior_checkpoint is not None:
            prior = torch.load(prior_checkpoint, map_location="cpu", weights_only=False)
            result["max_spring_log_change"] = float(
                (checkpoint["spring_Y"].log() - prior["spring_Y"].log()).abs().max()
            )
        (round_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        (output / "history.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"round {round_index}: {frames} frames, loss {result['loss']:.6g}, {previous}", flush=True)


if __name__ == "__main__":
    main()
