"""Heatmaps of the Fisher diagonal over the object, one column per action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asid.fim.case import load_case
from asid.fim.group import build_partition
from asid.fim.simulator import SimConfig
from asid.planning.fimmap import fisher_diagonal, group_values, plot_fim_heatmaps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--relative",
        action="store_true",
        help="subtract each action's mean, showing where the information sits",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run
    config = json.loads((run_dir / "config.json").read_text())
    gains = json.loads((run_dir / "gains.json").read_text())
    names = [action["name"] for action in gains["actions"]]

    scores = {action["name"]: action["gain"] for action in gains["actions"]}
    selected = [item["name"] for item in gains["selected"]]
    exact = (run_dir / "comparison.json").exists()
    if exact:
        comparison = json.loads((run_dir / "comparison.json").read_text())
        scores = comparison["exact"]
        selected = comparison["exact_selection"]

    case = load_case(
        config["case"], num_nodes=config["nodes"], max_frames=config["frames"]
    )
    partition = build_partition(
        case,
        config["groups"],
        seed=config["seed"],
        config=SimConfig(frames=config["frames"], num_substeps=config["substeps"]),
    )
    midpoints = case.object_midpoints

    diagonals = {}
    for name in names:
        directory = run_dir / "actions" / name
        matrix_path = directory / "fisher.npy"
        if matrix_path.exists():
            matrix = np.load(matrix_path)
        else:
            factors = np.load(directory / "U.npy")
            matrix = factors @ factors.T
        diagonals[name] = group_values(
            fisher_diagonal(matrix, config["groups"]),
            partition.spring_groups,
            partition.node_groups,
        )

    order = sorted(names, key=lambda name: scores[name], reverse=True)
    default = "fim_heatmaps_relative.png" if args.relative else "fim_heatmaps.png"
    figure = plot_fim_heatmaps(
        case.nodes,
        midpoints,
        diagonals,
        order,
        scores,
        args.out or run_dir / "figs" / default,
        title=(
            f"{case.name}  g={config['groups']}  frames={config['frames']}  "
            f"{'exact' if exact else 'rank-12 sketch'} F"
            + ("  |  log10 F minus per-action mean" if args.relative else "")
        ),
        selected=selected,
        resolution=args.resolution,
        relative=args.relative,
    )
    print(f"figure: {figure}")
    print(f"order: {[(name, round(scores[name], 1)) for name in order]}")


if __name__ == "__main__":
    main()
