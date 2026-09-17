"""Display a PhysTwin case depth sequence as a colorized video.

Examples:
    python tools/view_depth_video.py data/different_types/sim_soft_doll --camera 0
    python tools/view_depth_video.py data/different_types/sim_soft_doll \
        --camera 0 --output depth_cam0.mp4 --no_display
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def load_frames(case_dir: Path, camera: str) -> list[np.ndarray]:
    paths = sorted(
        (case_dir / "depth" / camera).glob("*.npy"),
        key=lambda path: int(path.stem),
    )
    if not paths:
        raise FileNotFoundError(f"No depth maps found in {case_dir / 'depth' / camera}")

    frames = []
    for path in paths:
        depth = np.load(path)
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2-D depth map: {path}")
        frames.append(depth.astype(np.float32) / 1000.0)
    return frames


def colorize(frames: list[np.ndarray], min_depth: float | None, max_depth: float | None):
    valid = np.concatenate(
        [frame[np.isfinite(frame) & (frame > 0)] for frame in frames]
    )
    if valid.size == 0:
        raise ValueError("Depth sequence contains no positive finite values")

    # MuJoCo commonly writes the camera far plane (often 30 m) into the
    # background. Exclude that background from automatic color scaling.
    scale_values = valid[valid < 10.0]
    if scale_values.size == 0:
        scale_values = valid
    low = min_depth if min_depth is not None else float(np.percentile(scale_values, 2))
    high = max_depth if max_depth is not None else float(np.percentile(scale_values, 98))
    if high <= low:
        raise ValueError(f"Invalid depth range: {low} to {high} metres")

    output = []
    for frame in frames:
        valid_pixels = np.isfinite(frame) & (frame > 0)
        normalized = np.clip((frame - low) / (high - low), 0, 1)
        image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[~valid_pixels] = 0
        output.append(image)
    return output, low, high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="PhysTwin case directory")
    parser.add_argument("--camera", default="0", help="Camera directory under depth/")
    parser.add_argument("--output", type=Path, help="Optional output MP4 path")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_depth", type=float, help="Color scale minimum in metres")
    parser.add_argument("--max_depth", type=float, help="Color scale maximum in metres")
    parser.add_argument("--no_display", action="store_true", help="Only write the video")
    args = parser.parse_args()

    if args.no_display and args.output is None:
        parser.error("--no_display requires --output")
    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error(
            "No desktop display detected; run this command from a GUI session "
            "or use --output video.mp4 --no_display"
        )

    frames = load_frames(args.case, args.camera)
    images, low, high = colorize(frames, args.min_depth, args.max_depth)
    writer = None
    if args.output is not None:
        height, width = images[0].shape[:2]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create video: {args.output}")

    display_failed = False
    try:
        for image in images:
            if writer is not None:
                writer.write(image)
            if not args.no_display and not display_failed:
                try:
                    cv2.imshow("Depth", image)
                    if cv2.waitKey(max(1, round(1000 / args.fps))) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    if writer is None:
                        raise RuntimeError(
                            "OpenCV has no GUI support; use --output ... --no_display"
                        ) from exc
                    print("OpenCV has no GUI support; continuing with video output only.")
                    display_failed = True
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display and not display_failed:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    print(f"frames={len(frames)} range={low:.3f}-{high:.3f}m")
    if args.output is not None:
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
