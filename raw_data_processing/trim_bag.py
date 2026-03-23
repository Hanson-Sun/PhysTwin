#!/usr/bin/env python3
"""
Trim RealSense bag files to a specific frame range.

Works by seeking the input bag to the start timestamp and recording
to a new file — no external dependencies beyond pyrealsense2.

Usage:
    python trim_bag.py <input_bag> <output_bag> --start FRAME --end FRAME
    python trim_bag.py <input_bag> --info
"""

import argparse
import datetime
import sys
from pathlib import Path

import pyrealsense2 as rs


def _scan_timestamps(bag_path: str) -> list:
    """Return timestamps (ms) for every synchronized color+depth pair in the bag."""
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    timestamps = []
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
                timestamps.append(cf.get_timestamp())
    finally:
        pipeline.stop()

    return timestamps


def bag_info(bag_path: str) -> None:
    """Print basic information about a bag file."""
    print(f"Scanning {Path(bag_path).name} ...")
    timestamps = _scan_timestamps(bag_path)
    if not timestamps:
        print("No synchronized frame pairs found.")
        return
    duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
    print(f"  Synchronized pairs : {len(timestamps)}")
    print(f"  Duration           : {duration_s:.2f}s")
    print(f"  Approx. FPS        : {len(timestamps) / duration_s:.1f}")


def trim_bag(input_bag: str, output_bag: str, start_frame: int = 0, end_frame: int = -1) -> int:
    """
    Trim a bag file to [start_frame, end_frame] (inclusive, 0-indexed).

    Strategy:
      Pass 1 — scan timestamps of all synchronized pairs.
      Pass 2 — seek the playback device to the start timestamp and record
                to the output file until end_frame pairs have been written.
                pyrealsense2's record-to-file captures everything that flows
                through the pipeline, so simply reading frames is enough.

    Returns the number of frame pairs written.
    """
    # ── Pass 1: collect timestamps ────────────────────────────────────────────
    print(f"Scanning {Path(input_bag).name} ...")
    timestamps = _scan_timestamps(input_bag)
    total = len(timestamps)

    if total == 0:
        raise ValueError("No synchronized frame pairs found in bag file.")

    # Clamp and resolve frame range
    start_frame = max(0, min(start_frame, total - 1))
    end_frame = total - 1 if end_frame < 0 else max(start_frame, min(end_frame, total - 1))
    num_frames = end_frame - start_frame + 1

    # Seek offset is relative to the first frame, not Unix epoch
    seek_offset_ms = timestamps[start_frame] - timestamps[0]

    print(f"Total pairs   : {total}")
    print(f"Trimming      : frames {start_frame}–{end_frame} ({num_frames} pairs)")
    print(f"Seek offset   : {seek_offset_ms:.0f} ms")

    # ── Pass 2: seek + record ─────────────────────────────────────────────────
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(input_bag), repeat_playback=False)
    config.enable_record_to_file(str(output_bag))

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    playback.seek(datetime.timedelta(milliseconds=seek_offset_ms))

    written = 0
    try:
        while written < num_frames:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError as e:
                if "Timeout" in str(e) or "Frame didn't arrive" in str(e):
                    break
                raise
            cf = frameset.first_or_default(rs.stream.color)
            df = frameset.first_or_default(rs.stream.depth)
            if cf and df:
                written += 1
    finally:
        pipeline.stop()

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Trim RealSense bag files to a specific frame range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show bag info (frame count, duration, FPS)
  python trim_bag.py input.bag --info

  # Trim frames 100–500 into a new file
  python trim_bag.py input.bag output.bag --start 100 --end 500

  # Tip: use view_bag.py to find exact frame numbers first
  python view_bag.py input.bag
  python trim_bag.py input.bag trimmed.bag --start 45 --end 200
        """)

    parser.add_argument("input_bag", help="Input .bag file")
    parser.add_argument("output_bag", nargs="?", help="Output .bag file (required unless --info)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (default: last)")
    parser.add_argument("--info", action="store_true", help="Print bag info and exit")

    args = parser.parse_args()

    if not Path(args.input_bag).exists():
        print(f"Error: file not found: {args.input_bag}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.info:
            bag_info(args.input_bag)
        else:
            if not args.output_bag:
                print("Error: output_bag is required for trimming", file=sys.stderr)
                sys.exit(1)
            written = trim_bag(args.input_bag, args.output_bag, args.start, args.end)
            print(f"\n✓ Wrote {written} frame pairs → {args.output_bag}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()