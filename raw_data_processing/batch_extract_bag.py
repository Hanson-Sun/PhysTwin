#!/usr/bin/env python3
"""
Batch-extract RealSense bag files from one or more session directories.

Input layout (one session directory):
    <session_dir>/
        camera_<serial_0>.bag
        camera_<serial_1>.bag
        camera_<serial_2>.bag
        camera_<serial_0>.original.bag   ← ignored

Output layout:
    <output_dir>/<case_name>/
        rgb/   0/  <frame>.png ...   0.mp4
               1/  <frame>.png ...   1.mp4
        depth/ 0/  <frame>.npy ...
               1/  <frame>.npy ...

Cameras are assigned integer IDs (0, 1, 2, …) in alphabetical bag-filename
order. .original.bag files are always skipped.

Usage:
    python batch_extract.py <session_dir>  --output-dir <out>
    python batch_extract.py <root_dir>     --output-dir <out> --batch
    python batch_extract.py <root_dir>     --output-dir <out> --batch --workers 4
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_bag(bag_path: Path) -> tuple[list, list, list]:
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    profile.get_device().as_playback().set_real_time(False)

    color_frames, depth_frames, timestamps = [], [], []
    try:
        while True:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError as e:
                if "Timeout" in str(e) or "Frame didn't arrive" in str(e):
                    break
                raise
            cf = frameset.first_or_default(rs.stream.color)
            df = frameset.first_or_default(rs.stream.depth)
            if cf and df:
                color_frames.append(np.asanyarray(cf.get_data()).copy())
                depth_frames.append(
                    np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
                )
                timestamps.append(cf.get_timestamp())
    finally:
        pipeline.stop()

    return color_frames, depth_frames, timestamps


# ── Saving ────────────────────────────────────────────────────────────────────

def save_camera(color_frames: list, depth_frames: list,
                case_dir: Path, camera_id: int, fps: float) -> None:
    rgb_dir   = case_dir / "color"   / str(camera_id)
    depth_dir = case_dir / "depth" / str(camera_id)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    h, w = color_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(case_dir / "rgb" / f"{camera_id}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
    )
    for i, frame in enumerate(color_frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_dir / f"{i}.png"), bgr)
        writer.write(bgr)
    writer.release()

    for i, depth in enumerate(depth_frames):
        np.save(str(depth_dir / f"{i}.npy"), depth)


# ── Session processing (runs in worker process) ───────────────────────────────

def process_session(session_dir: Path, output_dir: Path,
                    case_name: str, fps: float) -> tuple[str, bool, str]:
    """
    Extract all non-.original bags in session_dir.
    Returns (case_name, success, message) — safe to call in a subprocess.
    """
    bags = sorted(
        p for p in session_dir.glob("*.bag")
        if not p.name.endswith(".original.bag")
    )
    if not bags:
        return case_name, False, "no .bag files found"

    case_dir = output_dir / case_name
    lines = [f"  {len(bags)} camera(s)"]

    for cam_id, bag_path in enumerate(bags):
        try:
            color_frames, depth_frames, _ = extract_bag(bag_path)
            if not color_frames:
                raise ValueError("no synchronized frame pairs found")
            save_camera(color_frames, depth_frames, case_dir, cam_id, fps)
            lines.append(f"  [{cam_id}] {bag_path.name}  {len(color_frames)} frames ✓")
        except Exception as e:
            lines.append(f"  [{cam_id}] {bag_path.name}  ✗ {e}")
            return case_name, False, "\n".join(lines)

    return case_name, True, "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract RealSense bag files into training-data layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_extract.py recording_20260321/ --output-dir extracted/
  python batch_extract.py raw/ --output-dir extracted/ --batch
  python batch_extract.py raw/ --output-dir extracted/ --batch --workers 4
        """)

    parser.add_argument("input_dir", help="Session dir, or root dir with --batch")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument("--case-name", help="Case name override (single-session only)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument("--batch", action="store_true",
                        help="Process every sub-directory as a separate session")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel worker processes (default: 2)")

    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        sessions = [(d, d.name) for d in sorted(input_dir.iterdir()) if d.is_dir()]
        if not sessions:
            print(f"No sub-directories found under {input_dir}")
            sys.exit(0)
        print(f"Found {len(sessions)} session(s)  |  workers: {args.workers}\n")
    else:
        sessions = [(input_dir, args.case_name or input_dir.name)]

    n_ok = n_failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_session, sd, output_dir, name, args.fps): name
            for sd, name in sessions
        }
        for future in as_completed(futures):
            case_name, ok, detail = future.result()
            status = "✓" if ok else "✗"
            print(f"{status} {case_name}\n{detail}\n")
            if ok:
                n_ok += 1
            else:
                n_failed += 1

    if args.batch:
        print(f"Done — {n_ok} succeeded, {n_failed} failed")

    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()