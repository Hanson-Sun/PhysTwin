#!/usr/bin/env python3
"""
Filesystem utilities for working with extracted RealSense case directories.
"""

import glob
import os
from pathlib import Path


def list_bag_files(directory: str = ".") -> list:
    """Recursively find all .bag files under a directory."""
    return sorted(glob.glob(os.path.join(directory, "**/*.bag"), recursive=True))


def get_case_info(case_dir: str) -> dict:
    """Return frame counts and video presence for an extracted case directory."""
    case_dir = str(case_dir)

    color_frames = sum(
        len(glob.glob(os.path.join(d, "*.png")))
        for d in glob.glob(os.path.join(case_dir, "color", "*"))
        if os.path.isdir(d)
    )
    depth_frames = sum(
        len(glob.glob(os.path.join(d, "*.npy")))
        for d in glob.glob(os.path.join(case_dir, "depth", "*"))
        if os.path.isdir(d)
    )
    has_video = bool(glob.glob(os.path.join(case_dir, "color", "*.mp4")))

    return {
        "path": case_dir,
        "color_frames": color_frames,
        "depth_frames": depth_frames,
        "has_video": has_video,
    }


def validate_extraction(case_dir: str) -> bool:
    """
    Validate that a case directory has the expected extraction structure.
    Returns True if valid, False otherwise.
    """
    case_dir = str(case_dir)
    errors = []

    color_frame_count = sum(
        len(glob.glob(os.path.join(d, "*.png")))
        for d in glob.glob(os.path.join(case_dir, "color", "*"))
        if os.path.isdir(d)
    )
    depth_frame_count = sum(
        len(glob.glob(os.path.join(d, "*.npy")))
        for d in glob.glob(os.path.join(case_dir, "depth", "*"))
        if os.path.isdir(d)
    )

    if color_frame_count == 0:
        errors.append("No color frames (.png) found")
    if depth_frame_count == 0:
        errors.append("No depth frames (.npy) found")
    if color_frame_count != depth_frame_count and color_frame_count > 0 and depth_frame_count > 0:
        errors.append(
            f"Frame count mismatch: {color_frame_count} color vs {depth_frame_count} depth"
        )
    if not glob.glob(os.path.join(case_dir, "color", "*.mp4")):
        errors.append("No MP4 video found")

    if errors:
        print(f"Validation failed for {case_dir}:")
        for e in errors:
            print(f"  ✗ {e}")
        return False

    print(f"✓ {case_dir}  ({color_frame_count} frames, video present)")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utilities for extracted RealSense case directories")
    subparsers = parser.add_subparsers(dest="command")

    list_p = subparsers.add_parser("list-bags", help="List all .bag files under a directory")
    list_p.add_argument("--directory", default=".", help="Root directory to search (default: .)")

    validate_p = subparsers.add_parser("validate", help="Validate an extracted case directory")
    validate_p.add_argument("case_dir", help="Path to case directory")

    info_p = subparsers.add_parser("info", help="Show frame counts for a case directory")
    info_p.add_argument("case_dir", help="Path to case directory")

    args = parser.parse_args()

    if args.command == "list-bags":
        bags = list_bag_files(args.directory)
        if bags:
            print(f"Found {len(bags)} bag file(s):")
            for b in bags:
                print(f"  {b}")
        else:
            print("No .bag files found.")

    elif args.command == "validate":
        validate_extraction(args.case_dir)

    elif args.command == "info":
        info = get_case_info(args.case_dir)
        print(f"Path         : {info['path']}")
        print(f"Color frames : {info['color_frames']}")
        print(f"Depth frames : {info['depth_frames']}")
        print(f"Has video    : {'✓' if info['has_video'] else '✗'}")

    else:
        parser.print_help()