#!/usr/bin/env python3
"""
Generate VDA depth maps from RGB numpy files.
For each {clip_id}_rgb.npy  →  {clip_id}_depth_vda.npy

Usage:
    python infer_vda_simple.py [--clip_id bottle_lift_single_cam0] [--visualize]
"""

import argparse, sys, cv2
from pathlib import Path

import numpy as np
import torch

vda_root = Path(__file__).parent.parent / "Video-Depth-Anything"
if not vda_root.exists():
    raise RuntimeError(f"Video-Depth-Anything not found at {vda_root}")
sys.path.insert(0, str(vda_root))
from video_depth_anything.video_depth import VideoDepthAnything


CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def colorize(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    norm = (np.clip(depth, lo, hi) - lo) / (hi - lo + 1e-6)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def load_model(encoder: str, metric: bool, device: str) -> VideoDepthAnything:
    model = VideoDepthAnything(**CONFIGS[encoder], metric=metric)
    name  = f"{'metric_' if metric else ''}video_depth_anything_{encoder}.pth"
    ckpt  = vda_root / "checkpoints" / name
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="./temporal_depth_training_data")
    p.add_argument("--clip_id",   default=None)
    p.add_argument("--encoder",   default="vitl", choices=["vits", "vitb", "vitl"])
    p.add_argument("--metric",    action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz_dir",   default="./depth_visualizations")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    rgb_files = sorted(data_dir.glob("*_rgb.npy"))
    if args.clip_id:
        rgb_files = [f for f in rgb_files if args.clip_id in f.name]
    if not rgb_files:
        print("No RGB files found."); return

    print(f"Loading VDA ({args.encoder}, metric={args.metric})...")
    model = load_model(args.encoder, args.metric, device)

    if args.visualize:
        Path(args.viz_dir).mkdir(parents=True, exist_ok=True)

    for rgb_path in rgb_files:
        clip_id  = rgb_path.stem.replace("_rgb", "")
        out_path = data_dir / f"{clip_id}_depth_vda.npy"

        if out_path.exists():
            print(f"Skipping {clip_id} (already done)"); continue

        print(f"\n── {clip_id} ──")
        rgb = np.load(rgb_path)
        print(f"  {rgb.shape} {rgb.dtype}")

        depths_list, _ = model.infer_video_depth(rgb, target_fps=30, input_size=518, device=device)
        depths = np.stack(depths_list).astype(np.float32)  # [T, H, W]

        np.save(out_path, depths)
        print(f"  Saved {out_path}  shape={depths.shape}")

        if args.visualize:
            T, H, W = depths.shape
            vp = Path(args.viz_dir) / f"{clip_id}_depth_vis.mp4"
            vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
            for i in range(T): vw.write(colorize(depths[i]))
            vw.release()
            print(f"  Saved {vp}")

    print("\nDone!")


if __name__ == "__main__":
    main()