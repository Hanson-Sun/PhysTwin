"""Convert paired RGB-D videos into a PhysTwin processed case."""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        with path.open("r", encoding="utf-8") as file:
            matrix = np.asarray(json.load(file), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def prepare_case(args: argparse.Namespace) -> Path:
    case_dir = Path(args.output_dir) / args.case_name
    color_dir = case_dir / "color" / "0"
    depth_dir = case_dir / "depth" / "0"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    color_capture = cv2.VideoCapture(str(args.rgb_video))
    depth_capture = cv2.VideoCapture(str(args.depth_video))
    if not color_capture.isOpened() or not depth_capture.isOpened():
        raise RuntimeError("Unable to open both RGB and depth videos")

    rgb_fps = color_capture.get(cv2.CAP_PROP_FPS)
    depth_fps = depth_capture.get(cv2.CAP_PROP_FPS)
    fps = args.fps or rgb_fps or depth_fps or 30.0
    if rgb_fps and depth_fps and abs(rgb_fps - depth_fps) > 0.5:
        raise ValueError(f"RGB/depth FPS mismatch: {rgb_fps:.3f} vs {depth_fps:.3f}")

    writer = None
    frame_count = 0
    try:
        while True:
            rgb_ok, rgb_frame = color_capture.read()
            depth_ok, depth_frame = depth_capture.read()
            if not rgb_ok or not depth_ok:
                if rgb_ok != depth_ok:
                    raise ValueError("RGB and depth videos contain different frame counts")
                break

            if depth_frame.ndim == 3:
                depth_frame = depth_frame[:, :, 0]
            if rgb_frame.shape[:2] != depth_frame.shape[:2]:
                raise ValueError("RGB and depth frame dimensions do not match")

            if writer is None:
                height, width = rgb_frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(case_dir / "color" / "0.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Unable to create normalized RGB video")

            frame_name = f"{frame_count}.png"
            cv2.imwrite(str(color_dir / frame_name), rgb_frame)
            depth_mm = depth_frame.astype(np.float32) * args.depth_scale
            np.save(depth_dir / f"{frame_count}.npy", depth_mm)
            writer.write(rgb_frame)
            frame_count += 1
    finally:
        color_capture.release()
        depth_capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0 or writer is None:
        raise ValueError("The input videos contained no paired frames")

    intrinsics = load_matrix(args.intrinsics)
    c2w = load_matrix(args.c2w)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got {intrinsics.shape}")
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {c2w.shape}")

    height, width = cv2.imread(str(color_dir / "0.png")).shape[:2]
    with (case_dir / "calibrate.pkl").open("wb") as file:
        pickle.dump([c2w], file)
    with (case_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "intrinsics": [intrinsics.tolist()],
                "WH": [width, height],
                "fps": float(fps),
                "num_frames": frame_count,
                "depth_scale": args.depth_scale,
            },
            file,
            indent=2,
        )
    return case_dir


def run_processing(args: argparse.Namespace, case_dir: Path) -> None:
    command = [
        sys.executable,
        "process_data.py",
        "--base_path",
        str(case_dir.parent),
        "--case_name",
        case_dir.name,
        "--category",
        args.category,
    ]
    if args.shape_prior:
        command.append("--shape_prior")
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb_video", type=Path, required=True)
    parser.add_argument("--depth_video", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--c2w", type=Path, required=True)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Multiplier converting the depth video values to millimetres.",
    )
    parser.add_argument("--shape_prior", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = prepare_case(args)
    print(f"Prepared {case_dir}")
    if not args.prepare_only:
        run_processing(args, case_dir)


if __name__ == "__main__":
    main()