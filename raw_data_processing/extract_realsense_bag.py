#!/usr/bin/env python3
"""
Extract a RealSense bag file into a structured directory for training pipelines.

Output layout:
    <output_dir>/<case_name>/color/<camera_id>/<frame_idx>.png
    <output_dir>/<case_name>/color/<camera_id>.mp4
    <output_dir>/<case_name>/depth/<camera_id>/<frame_idx>.npy   (float32, metres)

Usage:
    python extract_realsense_bag.py <bag_file> <case_name> [options]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_bag(bag_path: str) -> tuple[list, list, list]:
    """
    Extract all synchronized color+depth pairs from a RealSense bag file.

    Returns:
        color_frames  — list of uint8 RGB arrays (H, W, 3)
        depth_frames  — list of float32 depth arrays in metres (H, W)
        timestamps    — list of timestamps in milliseconds
    """
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
                # .copy() is critical — without it the array is a view into the
                # camera buffer which gets invalidated when the pipeline advances.
                color_frames.append(np.asanyarray(cf.get_data()).copy())
                depth_frames.append(
                    np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
                )
                timestamps.append(cf.get_timestamp())
    finally:
        pipeline.stop()

    if not color_frames:
        raise ValueError("No synchronized frame pairs found in bag file.")

    print(f"Extracted {len(color_frames)} frame pairs")
    return color_frames, depth_frames, timestamps


# ── Saving ────────────────────────────────────────────────────────────────────

def save_frames_and_video(
    color_frames: list,
    depth_frames: list,
    output_dir: str,
    case_name: str,
    camera_id: str,
    fps: float = 30.0,
) -> None:
    """Save color frames as PNGs + MP4, depth as NPY files."""
    color_dir = Path(output_dir) / case_name / "color" / camera_id
    depth_dir = Path(output_dir) / case_name / "depth" / camera_id
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    h, w = color_frames[0].shape[:2]

    # VideoWriter opened once, reused for both PNG and video to avoid
    # converting each frame twice.
    video_path = Path(output_dir) / case_name / "color" / f"{camera_id}.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}. ")

    print(f"Saving color frames → {color_dir}")
    for i, frame in enumerate(color_frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(color_dir / f"{i}.png"), bgr)
        writer.write(bgr)

    writer.release()
    print(f"  {len(color_frames)} PNGs + MP4 written")

    print(f"Saving depth frames → {depth_dir}")
    for i, depth in enumerate(depth_frames):
        np.save(str(depth_dir / f"{i}.npy"), depth)
    print(f"  {len(depth_frames)} NPY files written")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract a RealSense bag into a training-data directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_realsense_bag.py recording.bag my_case --camera-id 0
  python extract_realsense_bag.py recording.bag my_case --output-dir /data --fps 15
        """)

    parser.add_argument("bag_file", help="Path to .bag file")
    parser.add_argument("case_name", help="Case name used to organise output directories")
    parser.add_argument("--camera-id", default="0", help="Camera ID label (default: 0)")
    parser.add_argument("--output-dir", default="data/different_types",
                        help="Root output directory (default: data/different_types)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate for the output video (default: 30)")

    args = parser.parse_args()

    if not Path(args.bag_file).exists():
        print(f"Error: file not found: {args.bag_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Bag file   : {args.bag_file}")
    print(f"Case       : {args.case_name}")
    print(f"Camera ID  : {args.camera_id}")
    print(f"Output dir : {args.output_dir}")
    print(f"Video FPS  : {args.fps}\n")

    color_frames, depth_frames, _ = extract_bag(args.bag_file)
    save_frames_and_video(
        color_frames, depth_frames,
        args.output_dir, args.case_name, args.camera_id, args.fps,
    )
    out = Path(args.output_dir) / args.case_name
    print(f"\n✓ Done — output at {out}/")



if __name__ == "__main__":
    main()