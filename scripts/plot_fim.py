"""Redraw the FIM overlay from a saved run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asid.fim.group import group_indices
from asid.fim.viz import plot_fim_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run = args.run
    summary = json.loads((run / "summary.json").read_text())
    nodes = np.load(run / "nodes.npy")
    midpoints = np.load(run / "midpoints.npy")
    node_groups = np.load(run / "node_groups.npy")
    spring_groups = np.load(run / "spring_groups.npy")
    diag = np.load(run / "fim_diag.npy")
    groups = int(node_groups.max()) + 1
    indices = group_indices(groups)
    output = args.output or run / "figs" / "fim_partition.png"
    plot_fim_overlay(
        nodes,
        midpoints,
        node_groups,
        spring_groups,
        diag,
        indices,
        output,
        title=f"{summary['case']}  g={groups}  frames={summary['frames']}",
    )
    print(output)


if __name__ == "__main__":
    main()
