"""Run the complete PhysTwin warp-parameter training pipeline for one case."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--train_frame", type=int)
    parser.add_argument("--cma_max_iter", type=int, default=20)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--skip_cma", action="store_true")
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Save training videos every visualization interval (default).",
    )
    visualization.add_argument(
        "--no_visualize",
        dest="visualize",
        action="store_false",
        help="Disable saved training videos.",
    )
    parser.set_defaults(visualize=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.base_path / args.case_name
    split_path = case_dir / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing split: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    train_frame = args.train_frame or int(split["train"][1])

    common = [
        "--base_path",
        str(args.base_path),
        "--case_name",
        args.case_name,
        "--train_frame",
        str(train_frame),
    ]
    if not args.skip_cma:
        subprocess.run(
            [sys.executable, "optimize_cma.py", *common, "--max_iter", str(args.cma_max_iter)],
            check=True,
        )
    train_command = [sys.executable, "train_warp.py", *common]
    if args.iterations is not None:
        train_command.extend(["--iterations", str(args.iterations)])
    train_command.append("--visualize" if args.visualize else "--no_visualize")
    subprocess.run(train_command, check=True)

    train_dir = Path("experiments") / args.case_name / "train"
    checkpoints = list(train_dir.glob("best_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"Training completed without a best checkpoint in {train_dir}")
    best_checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    manifest = train_dir / "training_manifest.json"
    with manifest.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "case_name": args.case_name,
                "data_path": str(case_dir / "final_data.pkl"),
                "train_frame": train_frame,
                "best_checkpoint": str(best_checkpoint),
                "cma_enabled": not args.skip_cma,
            },
            file,
            indent=2,
        )
    print(f"Best warp checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()