"""Prepare and train a static Gaussian model for one processed PhysTwin case."""

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


EXPERIMENT_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
DEFAULT_PRUNED_ROOT = Path("gaussian_output_pruned_policy_30_55")
DEFAULT_RATIO_CSV = Path("benchmarks/pruning_ratio_policy_30_55.csv")


def copy_case_to_gaussian_source(case_dir: Path, source_dir: Path) -> None:
    metadata_path = case_dir / "metadata.json"
    calibration_path = case_dir / "calibrate.pkl"
    pcd_path = case_dir / "pcd" / "0.npz"
    masks_path = case_dir / "mask" / "processed_masks.pkl"
    if not metadata_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError(
            f"Missing calibration artifacts in {case_dir}; run RGB-D processing first"
        )
    if not pcd_path.is_file():
        raise FileNotFoundError(f"Missing processed point cloud: {pcd_path}")

    with calibration_path.open("rb") as file:
        c2ws = pickle.load(file)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    intrinsics = np.asarray(metadata["intrinsics"])
    if len(c2ws) != len(intrinsics):
        raise ValueError("Calibration poses and intrinsics have different view counts")

    source_dir.mkdir(parents=True, exist_ok=True)
    points_data = np.load(pcd_path)
    processed_masks = None
    if masks_path.is_file():
        with masks_path.open("rb") as file:
            processed_masks = pickle.load(file)

    all_points = []
    all_colors = []
    for view_index, c2w in enumerate(c2ws):
        image_path = case_dir / "color" / str(view_index) / "0.png"
        depth_path = case_dir / "depth" / str(view_index) / "0.npy"
        if not image_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"Missing RGB-D first frame for view {view_index}")
        shutil.copy2(image_path, source_dir / f"{view_index}.png")
        shutil.copy2(depth_path, source_dir / f"{view_index}_depth.npy")

        points = np.asarray(points_data["points"][view_index])
        colors = np.asarray(points_data["colors"][view_index])
        if processed_masks is not None:
            mask = np.asarray(processed_masks[0][view_index]["object"], dtype=bool)
            if len(mask) == len(points):
                points = points[mask]
                colors = colors[mask]
        all_points.append(points)
        all_colors.append(colors)

    with (source_dir / "camera_meta.pkl").open("wb") as file:
        pickle.dump({"c2ws": c2ws, "intrinsics": intrinsics}, file)
    with (source_dir / "interp_poses.pkl").open("wb") as file:
        pickle.dump([c2ws[0]], file)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.clip(np.vstack(all_colors), 0, 1))
    o3d.io.write_point_cloud(str(source_dir / "observation.ply"), point_cloud)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pruned_output_dir", type=Path, default=DEFAULT_PRUNED_ROOT)
    parser.add_argument("--pruning_ratio_csv", type=Path, default=DEFAULT_RATIO_CSV)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--init", choices=("pcd", "mesh", "hybrid"), default="hybrid")
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--use_high_res", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_prune", action="store_true")
    parser.add_argument("--force_prune", action="store_true")
    return parser


def read_keep_ratio(csv_path: Path, case_name: str) -> float:
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("case_name", "").strip() == case_name:
                    return float(row["keep_ratio"])
    return 0.30


def prune_gaussian(
    source_ply: Path,
    destination_ply: Path,
    keep_ratio: float,
    force: bool,
) -> None:
    if destination_ply.is_file() and not force:
        print(f"Using existing pruned Gaussian: {destination_ply}")
        return
    subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/prune_gaussians.py",
            "--input",
            str(source_ply),
            "--output",
            str(destination_ply),
            "--keep-ratio",
            str(keep_ratio),
            "--mode",
            "opacity_area",
        ],
        check=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    case_dir = args.base_path / args.case_name
    source_dir = args.output_dir / "source"
    model_dir = args.output_dir / EXPERIMENT_NAME
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_prepare:
        copy_case_to_gaussian_source(case_dir, source_dir)
    required = source_dir / "camera_meta.pkl"
    if not required.is_file():
        raise FileNotFoundError(f"Missing Gaussian source dataset: {required}")

    command = [
        sys.executable,
        "gs_train.py",
        "--source_path",
        str(source_dir),
        "--model_path",
        str(model_dir),
        "--gs_init_opt",
        args.init,
        "--iterations",
        str(args.iterations),
        "--save_iterations",
        str(args.iterations),
        "--disable_viewer",
    ]
    if args.use_masks:
        command.append("--use_masks")
    if args.use_high_res:
        command.append("--use_high_res")
    subprocess.run(command, check=True)
    point_cloud_path = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
    if not point_cloud_path.is_file():
        raise FileNotFoundError(f"Gaussian training completed without {point_cloud_path}")
    print(f"Gaussian model: {point_cloud_path}")
    if not args.skip_prune:
        keep_ratio = read_keep_ratio(args.pruning_ratio_csv, args.case_name)
        pruned_path = (
            args.pruned_output_dir
            / args.case_name
            / EXPERIMENT_NAME
            / "point_cloud"
            / f"iteration_{args.iterations}"
            / "point_cloud.ply"
        )
        prune_gaussian(point_cloud_path, pruned_path, keep_ratio, args.force_prune)
        print(f"Pruned Gaussian ({keep_ratio:.0%} kept): {pruned_path}")


if __name__ == "__main__":
    main()