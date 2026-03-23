#!/usr/bin/env python3
"""
Multi-camera RealSense bag file alignment and synchronization tool.

Loads all bag files and displays them stacked vertically for side-by-side
comparison. Use the S key to mark alignment points (saved as frame indices
per camera), then optionally export the config for use with trim_bag.py.

Usage:
    python align_multicam_bags.py --bags camera1.bag camera2.bag camera3.bag
    python align_multicam_bags.py --session /path/to/session/
    python align_multicam_bags.py --bags *.bag --output-config alignment.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _colorize_depth(depth_data: np.ndarray, depth_scale: float,
                    min_d_m: float, max_d_m: float) -> np.ndarray:
    """
    Convert a raw uint16 depth array to a colorized BGR image.

    Uses caller-supplied fixed bounds (in metres) so the colour mapping is
    stable across frames — per-frame normalization causes visible flickering
    even when the scene is static.
    """
    if max_d_m <= min_d_m:
        return np.zeros((*depth_data.shape, 3), dtype=np.uint8)

    depth_m = depth_data.astype(np.float32) * depth_scale
    normalized = np.zeros_like(depth_data, dtype=np.uint8)
    mask = depth_data > 0
    normalized[mask] = np.clip(
        (depth_m[mask] - min_d_m) / (max_d_m - min_d_m) * 255, 0, 255
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _load_bag(bag_path: str) -> Tuple[list, float, float, float]:
    """
    Load all synchronized color+depth pairs from a bag file.
    Frames are copied immediately so data outlives the pipeline.

    Returns:
        frames       — list of dicts with keys: color, depth, timestamp
        depth_scale  — sensor units → metres
        depth_min_m  — p1 of valid depth values across all frames (metres)
        depth_max_m  — p99 of valid depth values across all frames (metres)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    frames = []
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
                frames.append({
                    "color": np.asanyarray(cf.get_data()).copy(),
                    "depth": np.asanyarray(df.get_data()).copy(),
                    "timestamp": cf.get_timestamp(),
                })
    finally:
        pipeline.stop()

    # Compute global depth bounds in metres so colorization is flicker-free
    depth_min_m, depth_max_m = 0.0, 4.0
    if frames:
        all_valid = np.concatenate([
            f["depth"][f["depth"] > 0].astype(np.float32) * depth_scale
            for f in frames
            if np.any(f["depth"] > 0)
        ])
        if len(all_valid) > 0:
            depth_min_m = float(np.percentile(all_valid, 1))
            depth_max_m = float(np.percentile(all_valid, 99))

    return frames, depth_scale, depth_min_m, depth_max_m


def _build_timeline(frame_sets: Dict[str, list], max_gap_ms: float = 50.0) -> list:
    """
    Build a shared timeline of timestamp ticks from all cameras.

    Strategy:
      - Collect every timestamp from every camera.
      - Sort and deduplicate them with a minimum gap of max_gap_ms so that
        near-simultaneous frames from different cameras collapse to one tick.
      - For each tick, record the index of the closest frame in each camera
        (or None if that camera has no frame within max_gap_ms of the tick).

    Returns a list of dicts:
        {
            "tick_ms": float,
            "cam_indices": { cam_id: int | None, ... }
        }
    """
    if not frame_sets:
        return []

    all_ts: List[float] = []
    for frames in frame_sets.values():
        all_ts.extend(f["timestamp"] for f in frames)
    all_ts.sort()

    # Deduplicate into ticks separated by at least max_gap_ms
    ticks: List[float] = []
    for ts in all_ts:
        if not ticks or ts - ticks[-1] >= max_gap_ms:
            ticks.append(ts)

    # For each tick, find the nearest frame index per camera (O(n) pointer scan)
    timeline = []
    pointers: Dict[str, int] = {cam_id: 0 for cam_id in frame_sets}

    for tick in ticks:
        cam_indices: Dict[str, Optional[int]] = {}
        for cam_id, frames in frame_sets.items():
            if not frames:
                cam_indices[cam_id] = None
                continue
            p = pointers[cam_id]
            while p + 1 < len(frames) and (
                abs(frames[p + 1]["timestamp"] - tick)
                <= abs(frames[p]["timestamp"] - tick)
            ):
                p += 1
            pointers[cam_id] = p
            cam_indices[cam_id] = p if abs(frames[p]["timestamp"] - tick) <= max_gap_ms else None

        timeline.append({"tick_ms": tick, "cam_indices": cam_indices})

    return timeline


def _placeholder(height: int, width: int, label: str) -> np.ndarray:
    img = np.zeros((height, width * 2, 3), dtype=np.uint8)
    cv2.putText(img, label, (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    return img


# ── Aligner ───────────────────────────────────────────────────────────────────

class MultiCamAligner:
    _FALLBACK_H = 480
    _FALLBACK_W = 640

    def __init__(self, bag_files: List[str], fps: float = 30.0,
                 max_gap_ms: float = 50.0):
        self.fps = fps
        self.max_gap_ms = max_gap_ms
        self.bag_files = bag_files

        self.frame_sets: Dict[str, list] = {}
        self.bag_paths: Dict[str, str] = {}
        self.depth_scales: Dict[str, float] = {}
        self.depth_bounds: Dict[str, Tuple[float, float]] = {}  # cam_id → (min_m, max_m)

        print(f"Loading {len(bag_files)} camera(s)...")
        for bag_path in bag_files:
            cam_id = Path(bag_path).stem
            frames, depth_scale, depth_min_m, depth_max_m = _load_bag(bag_path)
            if not frames:
                print(f"  {cam_id}: 0 frames — skipping (bad recording?)")
                continue
            self.frame_sets[cam_id] = frames
            self.bag_paths[cam_id] = str(bag_path)
            self.depth_scales[cam_id] = depth_scale
            self.depth_bounds[cam_id] = (depth_min_m, depth_max_m)
            print(f"  {cam_id}: {len(frames)} frames  "
                  f"(ts {frames[0]['timestamp']:.0f} – {frames[-1]['timestamp']:.0f} ms)  "
                  f"depth {depth_min_m:.2f}–{depth_max_m:.2f}m")

        if not self.frame_sets:
            raise ValueError("No cameras with usable frames found.")

        self.timeline = _build_timeline(self.frame_sets, max_gap_ms)
        print(f"\nTimeline: {len(self.timeline)} ticks across {len(self.frame_sets)} camera(s)\n")

        first_frames = next(iter(self.frame_sets.values()))
        self._h, self._w = first_frames[0]["color"].shape[:2]

        self.tick_index = 0
        self.playing = False

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_tick(self, tick_idx: int) -> np.ndarray:
        tick = self.timeline[tick_idx]
        rows = []
        for cam_id in sorted(self.frame_sets):
            fi = tick["cam_indices"].get(cam_id)
            if fi is None:
                row = _placeholder(self._h, self._w, f"{cam_id}  [no frame]")
            else:
                frame = self.frame_sets[cam_id][fi]
                color_bgr = cv2.cvtColor(frame["color"], cv2.COLOR_RGB2BGR)

                min_m, max_m = self.depth_bounds[cam_id]
                depth_col = _colorize_depth(
                    frame["depth"], self.depth_scales[cam_id], min_m, max_m
                )

                ts_offset = frame["timestamp"] - self.timeline[0]["tick_ms"]
                cv2.putText(
                    color_bgr,
                    f"{cam_id}  f{fi + 1}/{len(self.frame_sets[cam_id])}  "
                    f"t={ts_offset / 1000:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                row = np.hstack([color_bgr, depth_col])
            rows.append(row)

        display = np.vstack(rows)
        status = "PLAYING" if self.playing else "PAUSED"
        tick_ms = tick["tick_ms"] - self.timeline[0]["tick_ms"]
        cv2.putText(
            display,
            f"{status}  tick {tick_idx + 1}/{len(self.timeline)}  "
            f"t={tick_ms / 1000:.2f}s  |  S:save  Q:quit",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1,
        )
        return display

    # ── Navigation ────────────────────────────────────────────────────────────

    def _clamp(self, idx: int) -> int:
        return max(0, min(idx, len(self.timeline) - 1))

    # ── Interactive loop ──────────────────────────────────────────────────────

    def run_interactive(self) -> List[Dict]:
        window = f"Multi-Camera Aligner  ({len(self.frame_sets)} cameras  |  {len(self.timeline)} ticks)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        print("Controls:")
        print("  SPACE       Play/Pause")
        print("  LEFT/RIGHT  Step ±1 tick")
        print("  UP/DOWN     Skip ±10 ticks")
        print("  R           Reset to first tick")
        print("  S           Save current tick as alignment point")
        print("  Q / ESC     Quit")
        print()

        alignment_points: List[Dict] = []

        try:
            while True:
                cv2.imshow(window, self._render_tick(self.tick_index))
                wait_ms = int(1000 / self.fps) if self.playing else 0
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    self.playing = not self.playing
                elif key == 83:    # RIGHT
                    self.playing = False
                    self.tick_index = self._clamp(self.tick_index + 1)
                elif key == 81:    # LEFT
                    self.playing = False
                    self.tick_index = self._clamp(self.tick_index - 1)
                elif key == 82:    # UP
                    self.playing = False
                    self.tick_index = self._clamp(self.tick_index + 10)
                elif key == 84:    # DOWN
                    self.playing = False
                    self.tick_index = self._clamp(self.tick_index - 10)
                elif key in (ord("r"), ord("R")):
                    self.playing = False
                    self.tick_index = 0
                elif key == ord("s"):
                    tick = self.timeline[self.tick_index]
                    point = {
                        "tick_ms": tick["tick_ms"],
                        "cam_indices": tick["cam_indices"],
                    }
                    alignment_points.append(point)
                    print(f"✓ Alignment point saved at tick {self.tick_index + 1}: "
                          f"{tick['cam_indices']}")
                elif self.playing:
                    if self.tick_index < len(self.timeline) - 1:
                        self.tick_index += 1
                    else:
                        self.playing = False
        finally:
            cv2.destroyAllWindows()

        return alignment_points

    # ── Export ────────────────────────────────────────────────────────────────

    def export_alignment_config(self, output_file: str,
                                alignment_points: List[Dict]) -> None:
        data = {
            "cameras": sorted(self.frame_sets.keys()),
            "bag_paths": self.bag_paths,
            "num_cameras": len(self.frame_sets),
            "max_gap_ms": self.max_gap_ms,
            "fps": self.fps,
            "alignment_points": alignment_points,
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Alignment config saved to: {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def find_bags_in_session(session_dir: str) -> List[str]:
    return [str(b) for b in sorted(Path(session_dir).glob("camera_*.bag"))]


def _default_config_path(bags: List[str], session_dir: Optional[str]) -> Path:
    if session_dir:
        return Path(session_dir) / "alignment.json"
    return Path(bags[0]).parent / "alignment.json"


def main():
    parser = argparse.ArgumentParser(
        description="Align multiple RealSense bag files by timestamp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python align_multicam_bags.py --session recording_20240101_120000/
  python align_multicam_bags.py --bags cam1.bag cam2.bag cam3.bag
  python align_multicam_bags.py --bags *.bag --output-config alignment.json
        """)

    parser.add_argument("--bags", nargs="+", help="Bag file paths")
    parser.add_argument("--session", help="Session directory with camera_*.bag files")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    parser.add_argument("--max-gap-ms", type=float, default=50.0,
                        help="Max timestamp gap (ms) to consider frames aligned (default: 50)")
    parser.add_argument("--output-config",
                        help="Where to save alignment JSON (default: <session_dir>/alignment.json)")

    args = parser.parse_args()

    if args.bags:
        bags = args.bags
    elif args.session:
        bags = find_bags_in_session(args.session)
    else:
        parser.print_help()
        sys.exit(1)

    if not bags:
        print("Error: no bag files found", file=sys.stderr)
        sys.exit(1)

    missing = [b for b in bags if not Path(b).exists()]
    if missing:
        for b in missing:
            print(f"Error: file not found: {b}", file=sys.stderr)
        sys.exit(1)

    try:
        aligner = MultiCamAligner(bags, fps=args.fps, max_gap_ms=args.max_gap_ms)
        alignment_points = aligner.run_interactive()

        if alignment_points:
            config_path = args.output_config or str(_default_config_path(bags, args.session))
            aligner.export_alignment_config(config_path, alignment_points)
        else:
            print("No alignment points saved — config not written.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()