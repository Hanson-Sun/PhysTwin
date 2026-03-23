#!/usr/bin/env python3
"""
Auto-trim all bag files found under a root directory.

Walks every subdirectory of <root_dir>. For each folder that contains an
alignment.json (produced by align_multicam_bags.py), trims the bag files
to the wall-clock window defined by the two saved alignment points.

Trimming is done by timestamp, not frame index, so all cameras produce the
same duration and frame counts stay consistent across cameras.

pyrealsense2 does not allow opening a bag for playback and recording at the
same time. The workaround:
  Pass 1  Read frames within the timestamp window into memory.
  Pass 2  Inject them into a software_device wrapped in rs.recorder.

For each camera:
  1. The original bag is renamed to <n>.original.bag  (backup)
  2. The trimmed bag is written to a temp file, then moved to the original path

If a .original.bag already exists the script skips that camera, so re-running
is always safe.

Usage:
    python auto_trim.py <root_dir>
    python auto_trim.py <root_dir> --dry-run
    python auto_trim.py <root_dir> --config-name my_alignment.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


# ── Pass 1: read frames by timestamp window ───────────────────────────────────

def _read_frame_window(bag_path: str, start_ms: float, end_ms: float) -> tuple:
    """
    Read all synchronized color+depth pairs whose timestamp falls within
    [start_ms, end_ms] from a playback bag into RAM.

    Using timestamps instead of frame indices ensures all cameras produce the
    same wall-clock duration regardless of per-camera frame-rate drift.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    c_w = c_h = c_fps = d_w = d_h = d_fps = 0
    c_fmt = rs.format.bgr8
    c_intrinsics = d_intrinsics = None

    for sp in profile.get_streams():
        vsp = sp.as_video_stream_profile()
        if sp.stream_type() == rs.stream.color:
            c_w, c_h, c_fps = vsp.width(), vsp.height(), vsp.fps()
            c_fmt = vsp.format()
            c_intrinsics = vsp.get_intrinsics()
        elif sp.stream_type() == rs.stream.depth:
            d_w, d_h, d_fps = vsp.width(), vsp.height(), vsp.fps()
            d_intrinsics = vsp.get_intrinsics()

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
            if not (cf and df):
                continue

            ts = cf.get_timestamp()

            # Skip frames before the window; stop once we're past it
            if ts < start_ms:
                continue
            if ts > end_ms:
                break

            color_frames.append(np.asanyarray(cf.get_data()).copy())
            depth_frames.append(np.asanyarray(df.get_data()).copy())
            timestamps.append(ts)
    finally:
        pipeline.stop()

    return (color_frames, depth_frames, timestamps,
            c_intrinsics, d_intrinsics,
            c_w, c_h, c_fps, c_fmt,
            d_w, d_h, d_fps,
            depth_scale)


# ── Pass 2: write in-memory frames to a new bag via software_device ───────────

def _write_frames_to_bag(output_bag: str,
                         color_frames: list, depth_frames: list,
                         timestamps: list,
                         c_intrinsics, d_intrinsics,
                         c_w, c_h, c_fps, c_fmt,
                         d_w, d_h, d_fps,
                         depth_scale: float) -> int:
    sw = rs.software_device()
    sensor = sw.add_sensor("Synthetic")

    cvs = rs.video_stream()
    cvs.type = rs.stream.color; cvs.index = 0; cvs.uid = 0
    cvs.width = c_w; cvs.height = c_h; cvs.fps = c_fps; cvs.fmt = c_fmt
    cvs.bpp = 3 if c_fmt in (rs.format.rgb8, rs.format.bgr8) else 4
    cvs.intrinsics = c_intrinsics
    color_sp = sensor.add_video_stream(cvs)

    dvs = rs.video_stream()
    dvs.type = rs.stream.depth; dvs.index = 0; dvs.uid = 1
    dvs.width = d_w; dvs.height = d_h; dvs.fps = d_fps
    dvs.fmt = rs.format.z16; dvs.bpp = 2
    dvs.intrinsics = d_intrinsics
    depth_sp = sensor.add_video_stream(dvs)

    sensor.add_read_only_option(rs.option.depth_units, depth_scale)

    recorder = rs.recorder(str(output_bag), sw)
    sensor.open([color_sp, depth_sp])
    sensor.start(lambda _: None)

    bpp_c = 3 if c_fmt in (rs.format.rgb8, rs.format.bgr8) else 4
    base_ts = timestamps[0] if timestamps else 0.0

    for i, (color, depth, ts) in enumerate(zip(color_frames, depth_frames, timestamps)):
        rel_ts = ts - base_ts

        cf = rs.software_video_frame()
        cf.pixels = color.tobytes(); cf.stride = c_w * bpp_c; cf.bpp = bpp_c
        cf.frame_number = i; cf.timestamp = rel_ts
        cf.domain = rs.timestamp_domain.system_time
        cf.profile = color_sp.as_video_stream_profile()
        sensor.on_video_frame(cf)

        df_sw = rs.software_video_frame()
        df_sw.pixels = depth.tobytes(); df_sw.stride = d_w * 2; df_sw.bpp = 2
        df_sw.depth_units = depth_scale; df_sw.frame_number = i
        df_sw.timestamp = rel_ts; df_sw.domain = rs.timestamp_domain.system_time
        df_sw.profile = depth_sp.as_video_stream_profile()
        sensor.on_video_frame(df_sw)

    sensor.stop()
    sensor.close()
    del recorder  # flushes and closes the bag

    return len(color_frames)


# ── High-level trim ───────────────────────────────────────────────────────────

def _trim(input_bag: str, output_bag: str, start_ms: float, end_ms: float) -> int:
    result = _read_frame_window(input_bag, start_ms, end_ms)
    color_frames = result[0]
    if not color_frames:
        raise ValueError("No frames found in the timestamp window.")
    return _write_frames_to_bag(output_bag, *result)


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    required = {"cameras", "bag_paths", "alignment_points"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing keys {missing} — re-run align_multicam_bags.py.")
    if len(cfg["alignment_points"]) < 2:
        raise ValueError(
            f"Need at least 2 alignment points (start + end), "
            f"got {len(cfg['alignment_points'])}."
        )
    return cfg


def _resolve_timestamp_window(cfg: dict) -> tuple[float, float]:
    """Extract the wall-clock trim window in ms from the two alignment points."""
    start_ms = cfg["alignment_points"][0]["tick_ms"]
    end_ms   = cfg["alignment_points"][1]["tick_ms"]
    if end_ms <= start_ms:
        raise ValueError(
            f"End tick ({end_ms:.0f} ms) must be after start tick ({start_ms:.0f} ms)."
        )
    return start_ms, end_ms


# ── Per-session processing ────────────────────────────────────────────────────

def _process_session(config_path: Path, dry_run: bool) -> tuple[int, int]:
    try:
        cfg = _load_config(config_path)
        start_ms, end_ms = _resolve_timestamp_window(cfg)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  ✗ Could not load config: {e}")
        return 0, 1

    duration_s = (end_ms - start_ms) / 1000.0
    print(f"  {len(cfg['cameras'])} camera(s)  |  "
          f"{start_ms:.0f}–{end_ms:.0f} ms  ({duration_s:.2f}s)")

    n_ok, n_failed = 0, 0

    for cam_id in cfg["cameras"]:
        bag_path = Path(cfg["bag_paths"].get(cam_id, ""))
        print(f"  ── {cam_id}")

        if not bag_path.exists():
            print(f"       ✗ Bag not found: {bag_path}")
            n_failed += 1
            continue

        backup_path = bag_path.with_suffix(".original.bag")
        tmp_path    = bag_path.with_suffix(".trimmed.bag")

        if backup_path.exists():
            print(f"       ✗ Backup already exists ({backup_path.name}) — skipping")
            n_failed += 1
            continue

        if dry_run:
            print(f"       [dry-run] trim {start_ms:.0f}–{end_ms:.0f} ms ({duration_s:.2f}s)")
            n_ok += 1
            continue

        try:
            written = _trim(str(bag_path), str(tmp_path), start_ms, end_ms)
            os.rename(bag_path, backup_path)
            os.rename(tmp_path, bag_path)
            print(f"       ✓ {written} pairs written  (backup: {backup_path.name})")
            n_ok += 1
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"       ✗ Failed: {e}")
            n_failed += 1

    return n_ok, n_failed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-trim RealSense bag files using alignment configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Record  →  <root>/<session>/camera_*.bag
  2. Align   →  <root>/<session>/alignment.json   (via align_multicam_bags.py)
  3. Trim:
       python auto_trim.py <root>
       python auto_trim.py <root> --dry-run        # preview first

Each original bag is preserved as <n>.original.bag.
Sessions without an alignment.json are silently skipped.
        """)

    parser.add_argument("root_dir", help="Root directory to search for session folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without modifying any files")
    parser.add_argument("--config-name", default="alignment.json",
                        help="Alignment config filename (default: alignment.json)")

    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    configs = sorted(root.rglob(args.config_name))
    if not configs:
        print(f"No '{args.config_name}' files found under {root}")
        sys.exit(0)

    print(f"Found {len(configs)} session(s) under {root}")
    if args.dry_run:
        print("(dry-run — no files will be modified)")
    print()

    total_ok, total_failed = 0, 0

    for config_path in configs:
        print(f"Session: {config_path.parent.relative_to(root)}")
        n_ok, n_failed = _process_session(config_path, args.dry_run)
        total_ok     += n_ok
        total_failed += n_failed
        print()

    print(f"Done  —  {total_ok} camera(s) trimmed,  {total_failed} skipped/failed")
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()