"""
calibrate_with_dust3r.py

Use DUSt3R to get jointly-consistent camera intrinsics and extrinsics from
a single static frame (one image per camera). Updates calibrate.pkl and
metadata.json in-place.

Usage:
  python calibrate_with_dust3r.py --case_dir data/different_types/cloth_double_bend
  python calibrate_with_dust3r.py --case_dir ... --frame 0 --weights /path/to/model.pth --backup

If --weights is not given, downloads "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
from HuggingFace (requires internet).
"""

import os
import sys
import json
import pickle
import shutil
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import PIL

# DUSt3R lives in dust3r/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dust3r"))

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_metadata(case_dir):
    path = os.path.join(case_dir, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_calibration(case_dir, c2ws, Ks, meta):
    calib_path = os.path.join(case_dir, "calibrate.pkl")
    with open(calib_path, "wb") as f:
        pickle.dump(c2ws, f)
    meta["intrinsics"] = [K.tolist() for K in Ks]
    with open(os.path.join(case_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {calib_path}")
    print(f"Saved metadata.json")


def get_frame_paths(case_dir, frame):
    """Return image paths for each camera at the given frame, in cam order."""
    color_dir = os.path.join(case_dir, "color")
    cam_dirs = sorted(
        [d for d in os.listdir(color_dir) if os.path.isdir(os.path.join(color_dir, d))],
        key=lambda x: int(x)
    )
    paths = []
    for cam in cam_dirs:
        p = os.path.join(color_dir, cam, f"{frame}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Frame {frame} not found for cam {cam}: {p}")
        paths.append(p)
    return paths


# ── scale K back to original resolution ──────────────────────────────────────
def rescale_intrinsics(K_dust3r, dust3r_hw, orig_hw):
    """
    DUSt3R resizes images internally. Rescale K from DUSt3R's resolution
    back to the original image resolution.
    K_dust3r: (3,3) at dust3r_hw = (H_d, W_d)
    orig_hw:  (H_o, W_o) original resolution
    """
    H_d, W_d = dust3r_hw
    H_o, W_o = orig_hw

    sx = W_o / W_d
    sy = H_o / H_d

    K = K_dust3r.copy()
    K[0, 0] *= sx   # fx
    K[1, 1] *= sy   # fy
    K[0, 2] *= sx   # cx
    K[1, 2] *= sy   # cy
    return K

def calibrate(model, case_dir, frames=None, image_size=512, niter=300, device=None):
    """
    Deprecated: Use DUSt3RPoseFinder.infer_calibration() instead.
    
    This function is kept for backward compatibility only.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "depth_inference"))
    from DUSt3RPoseFinder import DUSt3RPoseFinder
    
    # Handle backward compatibility
    if frames is None:
        frames = [0]
    elif isinstance(frames, int):
        frames = [frames]
    
    # Create finder and perform inference on first frame
    # Note: This is a simplified wrapper - full multi-frame support should use DUSt3RPoseFinder directly
    finder = DUSt3RPoseFinder(model=model)
    
    # Get frame names matching the frame indices
    frame_names = [f"{f:03d}.png" for f in frames]
    
    intrinsics, extrinsics = finder.infer_calibration(
        case_dir=case_dir,
        frame_names=frame_names,
        calib_frames=len(frames)
    )
    
    # Convert back to cam-to-world for backward compatibility
    c2ws = np.linalg.inv(extrinsics)
    
    return intrinsics, [c2ws[i] for i in range(len(c2ws))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0],
                        help="Frame indices to use for calibration (space-separated, e.g., 0 10 20). Defaults to [0]")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to DUSt3R weights (.pth). If omitted, downloads from HuggingFace.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backup", action="store_true",
                        help="Backup calibrate.pkl and metadata.json before overwriting")
    args = parser.parse_args()

    case_dir = args.case_dir

    if args.backup:
        for name in ("calibrate.pkl", "metadata.json"):
            src = os.path.join(case_dir, name)
            if os.path.exists(src):
                dst = src.replace(".", "_before_dust3r.", 1)
                shutil.copy2(src, dst)
                print(f"Backed up → {dst}")

    print(f"Loading DUSt3R model on {args.device}...")
    if args.weights is not None:
        model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(args.device)
    else:
        model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(args.device)
    model.eval()

    # Import and use DUSt3RPoseFinder
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "depth_inference"))
    from DUSt3RPoseFinder import DUSt3RPoseFinder

    finder = DUSt3RPoseFinder(model=model)
    
    # Get all frame names from the first camera to determine available frames
    color_dir = os.path.join(case_dir, "color")
    cam_dirs = sorted(
        [d for d in os.listdir(color_dir) if os.path.isdir(os.path.join(color_dir, d))],
        key=lambda x: int(x)
    )
    first_cam_dir = os.path.join(color_dir, cam_dirs[0])
    all_frame_names = sorted([f for f in os.listdir(first_cam_dir) if f.endswith(".png")])
    
    # Run calibration
    intrinsics_array, extrinsics_array = finder.infer_calibration(
        case_dir=Path(case_dir),
        frame_names=all_frame_names,
        calib_frames=len(args.frames)
    )

    # Convert extrinsics (w2c) back to c2w for saving
    c2ws = [np.linalg.inv(extrinsics_array[i]) for i in range(len(extrinsics_array))]
    Ks_new = [intrinsics_array[i] for i in range(len(intrinsics_array))]

    # ── Report ────────────────────────────────────────────────────────────────
    meta = load_metadata(case_dir)
    n_cams = len(Ks_new)
    old_Ks = [np.asarray(k) for k in meta["intrinsics"]]
    try:
        with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
            old_c2ws = pickle.load(f)
    except FileNotFoundError:
        old_c2ws = [np.eye(4)] * n_cams

    print("\n=== DUSt3R calibration results ===")
    for i in range(n_cams):
        dfx = Ks_new[i][0, 0] - old_Ks[i][0, 0]
        dfy = Ks_new[i][1, 1] - old_Ks[i][1, 1]
        dt  = np.linalg.norm(c2ws[i][:3, 3] - np.asarray(old_c2ws[i])[:3, 3])
        print(f"  cam{i}:  fx={Ks_new[i][0,0]:.1f} (Δ{dfx:+.1f})  "
              f"fy={Ks_new[i][1,1]:.1f} (Δ{dfy:+.1f})  "
              f"cx={Ks_new[i][0,2]:.1f}  cy={Ks_new[i][1,2]:.1f}  "
              f"Δt={dt*100:.1f}cm")

    save_calibration(case_dir, c2ws, Ks_new, meta)
    print("\nDone. Re-run data_process_pcd.py and data_process_track.py to rebuild object_points.")


if __name__ == "__main__":
    main()
