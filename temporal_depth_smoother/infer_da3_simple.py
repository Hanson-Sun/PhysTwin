#!/usr/bin/env python3
"""
Generate DA3 depth maps from multi-camera RGB numpy files.

For each {clip_id}_cam{N}_rgb.npy  →  {clip_id}_cam{N}_depth_da3.npy

Static scene/cameras. Frame 0 is run freely so DA3 estimates multi-view
geometry; its predicted extrinsics are reused for all subsequent frames with
align_to_input_ext_scale=True to enforce consistent metric scale.

Usage:
    python infer_da3_simple.py [--clip_id bottle_lift_single] [--visualize] [--debug]
"""

import argparse
import cv2
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3


def to_uint8(f: np.ndarray) -> np.ndarray:
    if f.dtype == np.uint8: return f.copy()
    return (np.clip(f, 0, 1) * 255).astype(np.uint8) if f.dtype in (np.float32, np.float64) else f.astype(np.uint8)


def extract_depth(pred) -> np.ndarray:
    """Return prediction depth as float32 [N, H, W], handles any batch dims."""
    d = pred.depth
    if torch.is_tensor(d):      d = d.detach().cpu().float().numpy()
    elif isinstance(d, list):   d = np.stack([x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x) for x in d])
    else:                       d = d.astype(np.float32)
    while d.ndim > 3: d = d.squeeze(0)
    return d


def colorize(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    norm = (np.clip(depth, lo, hi) - lo) / (hi - lo + 1e-6)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


class DA3Inferencer:
    def __init__(self, model_name: str, process_res: int = 504, window_size: int = 3, overlap: int = 2):
        self.process_res = process_res
        self.window_size = window_size
        self.overlap = overlap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} ...")
        self.model = DepthAnything3.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def run(self, rgb_paths: list, num_frames: int, viz_paths: list = None) -> list:
        num_cams, T = len(rgb_paths), num_frames
        rgb = [np.load(p, mmap_mode='r') for p in rgb_paths]

        # ── Frame 0: free inference, extract extrinsics as scale anchor ──
        pred0 = self.model.inference(
            image=[to_uint8(rgb[c][0]) for c in range(num_cams)],
            process_res=self.process_res,
        )
        d0 = extract_depth(pred0)                          # [N, H, W]
        depth_h, depth_w = d0.shape[1], d0.shape[2]
        print(f"Depth resolution: {depth_h}x{depth_w}")

        # Reconstruct [N, 4, 4] extrinsics (DA3 strips bottom row on output)
        ext = pred0.extrinsics
        if torch.is_tensor(ext): ext = ext.detach().cpu().numpy()
        ext_3x4 = ext.reshape(num_cams, 3, 4)
        bottom  = np.tile([[[0., 0., 0., 1.]]], (num_cams, 1, 1))
        cam_ext = np.concatenate([ext_3x4, bottom], axis=1)  # [N, 4, 4]

        # Intrinsics — required by nested model's camera encoder for FOV encoding
        intr = pred0.intrinsics
        if torch.is_tensor(intr): intr = intr.detach().cpu().numpy()
        cam_intr = intr.reshape(num_cams, 3, 3)  # [N, 3, 3]

        depths = [np.zeros((T, depth_h, depth_w), dtype=np.float32) for _ in range(num_cams)]
        for c in range(num_cams): depths[c][0] = d0[c]
        del pred0, d0

        # ── cv2 windows + optional video writers ──
        wins = [f"Cam {c}" for c in range(num_cams)]
        for w in wins: cv2.namedWindow(w, cv2.WINDOW_AUTOSIZE)

        writers = []
        if viz_paths:
            for c, vp in enumerate(viz_paths):
                vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (depth_w, depth_h)) if vp else None
                writers.append(vw if (vw and vw.isOpened()) else None)
        else:
            writers = [None] * num_cams

        def show(t):
            for c in range(num_cams):
                vis = colorize(depths[c][t])
                cv2.imshow(wins[c], vis)
                if writers[c]: writers[c].write(vis)

        show(0)
        cv2.waitKey(1)

        # ── Frames 1..T-1: sliding window inference ──
        stride = self.window_size - self.overlap
        start_idx, chunk_idx = 1, 0
        pbar = tqdm(total=T - 1, desc="Frames", unit="frame")
        
        while start_idx < T:
            end_idx = min(start_idx + self.window_size, T)
            window_indices = list(range(start_idx, end_idx))
            
            # Flatten batch: for each frame, add all cameras
            all_images = []
            all_intrinsics = []
            all_extrinsics = []
            for idx in window_indices:
                for c in range(num_cams):
                    all_images.append(to_uint8(rgb[c][idx]))
                    all_intrinsics.append(cam_intr[c])
                    all_extrinsics.append(cam_ext[c])
            
            intrinsics_batch = np.stack(all_intrinsics, axis=0)
            extrinsics_batch = np.stack(all_extrinsics, axis=0)
            
            pred = self.model.inference(
                image=all_images,
                extrinsics=extrinsics_batch,
                intrinsics=intrinsics_batch,
                align_to_input_ext_scale=True,
                process_res=self.process_res,
            )
            
            # Skip overlap frames on non-first chunks
            is_first_chunk = (chunk_idx == 0)
            save_start_idx = 0 if is_first_chunk else self.overlap
            
            # Unpack results: indexed as frame_idx * num_cams + cam_idx
            for frame_i, idx in enumerate(window_indices):
                if not is_first_chunk and frame_i < self.overlap:
                    continue
                
                for c in range(num_cams):
                    depth_idx = frame_i * num_cams + c
                    d = pred.depth[depth_idx]
                    if torch.is_tensor(d):
                        d = d.detach().cpu().numpy()
                    depths[c][idx] = d
                
                show(idx)
                pbar.update(1)
                if cv2.waitKey(1) == 27: 
                    break
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_idx += stride
            chunk_idx += 1
        
        pbar.close()
        
        for vw in writers:
            if vw: vw.release()
        cv2.destroyAllWindows()
        return depths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="/mnt/d/DATA/phystwin/temporal_depth_training_data_v2")
    p.add_argument("--clip_id",     default=None)
    p.add_argument("--model_name",  default="depth-anything/DA3NESTED-GIANT-LARGE")
    p.add_argument("--process_res", default=504, type=int)
    p.add_argument("--window_size", default=3, type=int, help="Sliding window size for frame chunks")
    p.add_argument("--overlap",     default=2, type=int, help="Overlap between windows")
    p.add_argument("--visualize",   action="store_true")
    p.add_argument("--viz_dir",     default="./depth_visualizations")
    p.add_argument("--overwrite",   action="store_true", help="Overwrite existing depth maps")
    args = p.parse_args()

    data_dir = Path(args.data_dir)

    # Group files by clip
    clips = {}
    for f in data_dir.glob("*_rgb.npy"):
        parts = f.stem.replace("_rgb", "").rsplit("_cam", 1)
        if len(parts) == 2:
            clips.setdefault(parts[0], {})[int(parts[1])] = f

    if args.clip_id:
        clips = {k: v for k, v in clips.items() if args.clip_id in k}
    if not clips:
        print("No clips found."); return

    model = DA3Inferencer(args.model_name, args.process_res, args.window_size, args.overlap)
    if args.visualize: Path(args.viz_dir).mkdir(parents=True, exist_ok=True)

    for clip_id, cam_paths in sorted(clips.items()):
        print(f"\n── {clip_id} ──")
        sorted_cams = sorted(cam_paths.items())
        out_paths = [data_dir / f"{clip_id}_cam{c}_depth_da3.npy" for c, _ in sorted_cams]

        if all(p.exists() for p in out_paths) and not args.overwrite:
            print("  Already done, skipping."); continue

        rgb_paths = [p for _, p in sorted_cams]
        T = min(np.load(p, mmap_mode='r').shape[0] for p in rgb_paths)
        viz = [Path(args.viz_dir) / f"{clip_id}_cam{c}_depth_vis.mp4" if args.visualize else None
               for c, _ in sorted_cams]

        depths = model.run(rgb_paths, T, viz)
        for (c, _), d, op in zip(sorted_cams, depths, out_paths):
            np.save(op, d)
            print(f"  Saved cam{c}: {d.shape}")


if __name__ == "__main__":
    main()