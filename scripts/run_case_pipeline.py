"""Run preprocessing, warp training, and Gaussian reconstruction for one case."""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_PATH = REPO_ROOT / "data" / "different_types"
DEFAULT_CONFIG = REPO_ROOT / "data_config.csv"
DEFAULT_GAUSSIAN_ROOT = REPO_ROOT / "gaussian_output"


def case_config(config_path: Path, case_name: str) -> tuple[str | None, bool]:
    if not config_path.is_file():
        return None, False
    with config_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) >= 3 and row[0].strip() == case_name:
                return row[1].strip(), row[2].strip().lower() == "true"
    return None, False


def run(command: list[str], label: str, clean_data_environment: bool = False) -> None:
    print(f"\n=== {label} ===")
    environment = None
    if clean_data_environment:
        environment = os.environ.copy()
        environment.pop("CUDA_HOME", None)
        environment.pop("LD_LIBRARY_PATH", None)
        environment["CUDA_HOME"] = "/usr/local/cuda-12.1"
        environment["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.1/lib64"
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=environment)


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not produced: {path}")


def default_data_python() -> str:
    """Prefer the phystwin-data env's python for RGB-D processing (sam2 lives there)."""
    data_python = Path(sys.executable).parents[2] / "phystwin-data" / "bin" / "python"
    return str(data_python) if data_python.is_file() else sys.executable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--category")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data_python",
        default=default_data_python(),
        help="Python executable for RGB-D processing; defaults to the "
        "phystwin-data environment's python, falling back to the active interpreter.",
    )
    parser.add_argument("--shape_prior", action="store_true")
    parser.add_argument("--no_shape_prior", action="store_true")
    parser.add_argument("--cma_max_iter", type=int, default=20)
    parser.add_argument("--warp_iterations", type=int)
    parser.add_argument("--gaussian_iterations", type=int, default=10_000)
    parser.add_argument("--gaussian_output_dir", type=Path)
    parser.add_argument("--pruned_output_dir", type=Path)
    parser.add_argument("--skip_process", action="store_true")
    parser.add_argument("--skip_segmentation", action="store_true")
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Enable visualizations for all enabled stages (default).",
    )
    visualization.add_argument(
        "--no_visualize",
        dest="visualize",
        action="store_false",
        help="Disable preprocessing visualizations and saved Warp videos.",
    )
    parser.set_defaults(visualize=True)
    parser.add_argument("--skip_warp", action="store_true")
    parser.add_argument("--skip_gaussians", action="store_true")
    parser.add_argument("--skip_cma", action="store_true")
    parser.add_argument("--force_prune", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = args.base_path / args.case_name
    require(case_dir, "Case directory")

    configured_category, configured_shape_prior = case_config(args.config, args.case_name)
    category = args.category or configured_category
    if category is None and not args.skip_process:
        raise ValueError(
            f"No category found for {args.case_name}; pass --category explicitly"
        )
    if args.shape_prior and args.no_shape_prior:
        raise ValueError("--shape_prior and --no_shape_prior are mutually exclusive")

    use_shape_prior = configured_shape_prior
    if args.shape_prior:
        use_shape_prior = True
    if args.no_shape_prior:
        use_shape_prior = False

    warp_iterations = args.warp_iterations
    cma_max_iter = args.cma_max_iter
    gaussian_iterations = args.gaussian_iterations
    if args.smoke_test:
        warp_iterations = warp_iterations or 1
        cma_max_iter = 0
        gaussian_iterations = min(gaussian_iterations, 1)

    if args.dry_run:
        print(f"Case: {args.case_name}")
        print(f"Data python: {args.data_python}")
        print(f"Category: {category or '<not needed with --skip_process>'}")
        print(f"Shape prior: {use_shape_prior}")
        print(f"Process data: {not args.skip_process}")
        print(f"Train warp: {not args.skip_warp}")
        print(f"Reconstruct Gaussians: {not args.skip_gaussians}")
        print(f"CMA iterations: {cma_max_iter}")
        print(f"Warp iterations: {warp_iterations or '<config default>'}")
        print(f"Gaussian iterations: {gaussian_iterations}")
        print(f"Visualization: {args.visualize}")
        return

    if not args.skip_process:
        process_command = [
            *shlex.split(args.data_python),
            "process_data.py",
            "--base_path",
            str(args.base_path),
            "--case_name",
            args.case_name,
            "--category",
            category,
        ]
        if use_shape_prior:
            process_command.append("--shape_prior")
        if args.skip_segmentation:
            process_command.append("--skip_segmentation")
        if args.visualize:
            process_command.append("--visualize")
        run(process_command, "PhysTwin RGB-D processing", clean_data_environment=True)
    require(case_dir / "final_data.pkl", "Processed motion data")
    require(case_dir / "split.json", "Train/test split")

    if not args.skip_warp:
        warp_command = [
            sys.executable,
            "scripts/train_warp_case.py",
            "--base_path",
            str(args.base_path),
            "--case_name",
            args.case_name,
            "--cma_max_iter",
            str(cma_max_iter),
        ]
        if args.skip_cma or args.smoke_test:
            warp_command.append("--skip_cma")
        if warp_iterations is not None:
            warp_command.extend(["--iterations", str(warp_iterations)])
        warp_command.append("--visualize" if args.visualize else "--no_visualize")
        run(warp_command, "Warp parameter training")
        require(
            REPO_ROOT / "experiments" / args.case_name / "train" / "training_manifest.json",
            "Warp training manifest",
        )

    if not args.skip_gaussians:
        gaussian_output_dir = args.gaussian_output_dir or (
            DEFAULT_GAUSSIAN_ROOT / args.case_name
        )
        gaussian_command = [
            sys.executable,
            "scripts/reconstruct_gaussians.py",
            "--base_path",
            str(args.base_path),
            "--case_name",
            args.case_name,
            "--output_dir",
            str(gaussian_output_dir),
            "--iterations",
            str(gaussian_iterations),
            "--use_masks",
        ]
        if args.pruned_output_dir is not None:
            gaussian_command.extend(["--pruned_output_dir", str(args.pruned_output_dir)])
        if args.force_prune:
            gaussian_command.append("--force_prune")
        run(gaussian_command, "Static Gaussian reconstruction and pruning")
        require(
            gaussian_output_dir
            / "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
            / "point_cloud"
            / f"iteration_{gaussian_iterations}"
            / "point_cloud.ply",
            "Full Gaussian model",
        )

    print(f"\nPipeline complete: {args.case_name}")


if __name__ == "__main__":
    main()