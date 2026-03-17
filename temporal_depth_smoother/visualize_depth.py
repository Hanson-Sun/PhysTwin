#!/usr/bin/env python3
"""
Visualize depth sequences or masks from .npy files.
Usage: python visualize_depth.py <path.npy> [--fps 30]
"""
import argparse
import numpy as np
import cv2
from pathlib import Path


def colorize(frame: np.ndarray) -> np.ndarray:
    lo, hi = frame.min(), frame.max()
    norm = ((frame - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8) if hi > lo else np.zeros_like(frame, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask > 0.5] = (0, 255, 0)  # green = background
    vis[mask < 0.5] = (0, 0, 255)  # red   = foreground
    return vis


def visualize(npy_path: str, fps: int = 30):
    data = np.load(npy_path)
    name = Path(npy_path).name
    is_mask = 'mask' in name

    # Normalise to [T, H, W]
    if data.ndim == 2:
        data = data[np.newaxis]  # single frame
    elif data.ndim != 3:
        raise ValueError(f"Expected [H,W] or [T,H,W], got {data.shape}")

    T, H, W = data.shape
    print(f"{name}  {data.shape}  range=[{data.min():.4f}, {data.max():.4f}]")
    if is_mask:
        print(f"  Background: {(data[0] > 0.5).mean()*100:.1f}%  (green=bg, red=fg)")

    single = T == 1
    frame_idx, paused = 0, False
    print("Controls: SPACE=pause  ←/→=step  Q/ESC=quit" if not single else "Press any key to close")

    while True:
        frame = data[frame_idx]
        vis = colorize_mask(frame) if is_mask else colorize(frame)

        if not single:
            cv2.putText(vis, f"{frame_idx+1}/{T}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(name, vis)
        key = cv2.waitKey(0 if (paused or single) else int(1000 / fps))

        if key in (ord('q'), 27):   break
        elif key == ord(' '):       paused = not paused
        elif key in (83, ord('d')): frame_idx = (frame_idx + 1) % T
        elif key in (81, ord('a')): frame_idx = (frame_idx - 1) % T
        elif single:                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npy_path")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    visualize(args.npy_path, args.fps)