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


def calibrate(model, case_dir, frame=0, image_size=512, niter=300, device=None):
    """
    Run DUSt3R calibration on a single static frame.
    """
    if device is None:
        device = str(next(model.parameters()).device)

    img_paths = get_frame_paths(case_dir, frame)
    n_cams    = len(img_paths)

    print(f"Using frame {frame} from {n_cams} cameras:")
    for i, p in enumerate(img_paths):
        print(f"  cam{i}: {p}")

    imgs = load_images(img_paths, size=image_size, verbose=True)
    
    # use original image size for rescaling K, since DUSt3R operates on resized images internally
    W_orig, H_orig = PIL.Image.open(img_paths[0]).size
    
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"\nRunning inference on {len(pairs)} pairs...")
    output = inference(pairs, model, device, batch_size=1, verbose=True)

    print(f"\nRunning global alignment ({niter} iterations)...")
    mode = GlobalAlignerMode.PointCloudOptimizer if n_cams > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

    K_dust3r  = to_numpy(scene.get_intrinsics())   # (N, 3, 3) at DUSt3R's resolution
    c2ws_dust = to_numpy(scene.get_im_poses())      # (N, 4, 4) cam-to-world

    dust3r_H = imgs[0]["true_shape"][0][0].item() if hasattr(imgs[0]["true_shape"][0], "item") else int(imgs[0]["true_shape"][0][0])
    dust3r_W = imgs[0]["true_shape"][0][1].item() if hasattr(imgs[0]["true_shape"][0], "item") else int(imgs[0]["true_shape"][0][1])
    print(f"\nDUSt3R processed at: {dust3r_W}x{dust3r_H}")

    print(f"Original image size: {W_orig}x{H_orig}")
    Ks = [rescale_intrinsics(K_dust3r[i], (dust3r_H, dust3r_W), (H_orig, W_orig)) for i in range(n_cams)]

    c2ws = [c2ws_dust[i] for i in range(n_cams)]
    return Ks, c2ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to use for calibration (should be static/background)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to DUSt3R weights (.pth). If omitted, downloads from HuggingFace.")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224])
    parser.add_argument("--niter", type=int, default=300,
                        help="Global alignment iterations")
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

    Ks_new, c2ws_new = calibrate(
        model, case_dir,
        frame=args.frame, image_size=args.image_size, niter=args.niter, device=args.device,
    )

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
        dt  = np.linalg.norm(c2ws_new[i][:3, 3] - np.asarray(old_c2ws[i])[:3, 3])
        print(f"  cam{i}:  fx={Ks_new[i][0,0]:.1f} (Δ{dfx:+.1f})  "
              f"fy={Ks_new[i][1,1]:.1f} (Δ{dfy:+.1f})  "
              f"cx={Ks_new[i][0,2]:.1f}  cy={Ks_new[i][1,2]:.1f}  "
              f"Δt={dt*100:.1f}cm")

    save_calibration(case_dir, c2ws_new, Ks_new, meta)
    print("\nDone. Re-run data_process_pcd.py and data_process_track.py to rebuild object_points.")


if __name__ == "__main__":
    main()
