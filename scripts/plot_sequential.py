"""Redraw the sequential selection figure from a saved run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asid.planning.viz import plot_sequential_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads((args.run / "rounds.json").read_text())
    figure = plot_sequential_scores(
        payload["rounds"],
        args.out or args.run / "figs" / "sequential_scores.png",
        title=f"{args.run.name}  sequential  rank={payload['rank']}",
    )
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
