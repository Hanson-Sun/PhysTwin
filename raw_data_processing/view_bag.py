#!/usr/bin/env python3
"""
Interactive RealSense bag file viewer with frame-by-frame playback.

Controls:
  SPACE           Play/Pause
  LEFT/RIGHT      Previous/Next frame
  UP/DOWN         Skip ±10 frames
  R               Reset to first frame
  Q or ESC        Quit

Usage:
    python view_bag.py <bag_file> [--fps FPS]
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
import pyrealsense2 as rs


def colorize_depth(depth_data, depth_scale, min_d_m, max_d_m):
    """
    Convert a raw uint16 depth array to a colorized BGR image.

    Uses caller-supplied fixed bounds (in metres) so the colour mapping is
    stable across frames — per-frame normalization causes visible flickering
    even when the scene is static.

    Args:
        depth_data:  uint16 array of raw sensor values
        depth_scale: sensor units → metres conversion factor
        min_d_m:     lower bound in metres (maps to colour 0)
        max_d_m:     upper bound in metres (maps to colour 255)
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


def match_frames(color_list, depth_list, max_gap_ms=50):
    """
    Pair color and depth frames by nearest timestamp.
    Both lists are assumed to be in chronological order.
    Uses a linear scan (O(n)) instead of a nested loop.
    """
    pairs = []
    j = 0
    for cf in color_list:
        while j + 1 < len(depth_list) and (
            abs(depth_list[j + 1]['ts'] - cf['ts']) <= abs(depth_list[j]['ts'] - cf['ts'])
        ):
            j += 1
        if abs(depth_list[j]['ts'] - cf['ts']) < max_gap_ms:
            pairs.append({'color': cf['data'], 'depth': depth_list[j]['data'], 'ts': cf['ts']})
    return pairs


class BagViewer:
    def __init__(self, bag_path, fps=30.0):
        self.bag_path = Path(bag_path)
        self.frame_pause_ms = int(1000 / fps)
        self.frames = []
        self.frame_index = 0
        self.playing = False
        self.depth_scale = 1.0
        self.depth_min_m = 0.0
        self.depth_max_m = 4.0

        self._load_frames()
        if not self.frames:
            raise ValueError("No synchronized frame pairs found in bag file.")

    def _load_frames(self):
        print(f"Loading {self.bag_path.name} ...")

        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, str(self.bag_path), repeat_playback=False)
        profile = pipeline.start(config)

        # Depth scale converts raw uint16 → metres
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        streams = [f"{s.stream_type()} {s.format()} @ {s.fps()}fps" for s in profile.get_streams()]
        print(f"Streams: {', '.join(streams)}")

        color_list, depth_list = [], []

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

                if cf:
                    color_list.append({'data': np.asanyarray(cf.get_data()).copy(), 'ts': cf.get_timestamp()})
                if df:
                    depth_list.append({'data': np.asanyarray(df.get_data()).copy(), 'ts': df.get_timestamp()})
        finally:
            pipeline.stop()

        print(f"Captured {len(color_list)} color / {len(depth_list)} depth frames")

        self.frames = match_frames(color_list, depth_list)
        print(f"Matched {len(self.frames)} synchronized pairs")

        # Compute global depth bounds in metres across all frames so the
        # colour mapping is fixed and flicker-free.
        all_valid = np.concatenate([
            f['depth'][f['depth'] > 0].astype(np.float32) * self.depth_scale
            for f in self.frames
            if np.any(f['depth'] > 0)
        ])
        if len(all_valid) > 0:
            self.depth_min_m = float(np.percentile(all_valid, 1))
            self.depth_max_m = float(np.percentile(all_valid, 99))
        print(f"Depth range: {self.depth_min_m:.2f}m – {self.depth_max_m:.2f}m  (global, p1–p99)\n")

    def _render(self, idx):
        frame = self.frames[idx]
        color_bgr = cv2.cvtColor(frame['color'], cv2.COLOR_RGB2BGR)
        depth_colored = colorize_depth(
            frame['depth'], self.depth_scale, self.depth_min_m, self.depth_max_m
        )

        display = np.hstack([color_bgr, depth_colored])
        h = display.shape[0]

        status = "PLAYING" if self.playing else "PAUSED"
        cv2.putText(display,
                    f"Frame {idx + 1}/{len(self.frames)} | {status} | "
                    f"Depth: {self.depth_min_m:.2f}-{self.depth_max_m:.2f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE:Play/Pause  <->:Step  UP/DN:Skip10  R:Reset  Q:Quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display

    def run(self):
        window = f"RealSense Viewer — {self.bag_path.name}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        while True:
            cv2.imshow(window, self._render(self.frame_index))
            key = cv2.waitKey(self.frame_pause_ms if self.playing else 0) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == 83:  # RIGHT
                self.playing = False
                self.frame_index = min(self.frame_index + 1, len(self.frames) - 1)
            elif key == 81:  # LEFT
                self.playing = False
                self.frame_index = max(self.frame_index - 1, 0)
            elif key == 82:  # UP
                self.playing = False
                self.frame_index = min(self.frame_index + 10, len(self.frames) - 1)
            elif key == 84:  # DOWN
                self.playing = False
                self.frame_index = max(self.frame_index - 10, 0)
            elif key in (ord('r'), ord('R')):
                self.frame_index = 0
                self.playing = False
            elif self.playing:
                if self.frame_index < len(self.frames) - 1:
                    self.frame_index += 1
                else:
                    self.playing = False

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive RealSense bag viewer")
    parser.add_argument("bag_file", help="Path to .bag file")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    args = parser.parse_args()

    if not Path(args.bag_file).exists():
        print(f"Error: file not found: {args.bag_file}", file=sys.stderr)
        sys.exit(1)

    try:
        BagViewer(args.bag_file, fps=args.fps).run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()