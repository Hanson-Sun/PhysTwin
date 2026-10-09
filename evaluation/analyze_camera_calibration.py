"""
analyze_camera_calibration.py

Visualize whether per-camera DA3-estimated calibrations are mutually consistent
by checking if different cameras agree on world-space positions of the same cloth.

Produces:
  {case_dir}/calibration_analysis/per_camera_world_positions.png  -- scatter plot
  {case_dir}/calibration_analysis/per_camera_world_positions.mp4  -- video over time

Usage:
  python analyze_camera_calibration.py --case_dir data/different_types/cloth_double_bend
"""

import os
import pickle
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tqdm import tqdm


CAM_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_calibration(case_dir):
    with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)     # list of (4,4) arrays, one per camera
    with open(os.path.join(case_dir, "metadata.json")) as f:
        meta = json.load(f)
    Ks  = [np.array(k, dtype=np.float64) for k in meta["intrinsics"]]
    WH  = meta["WH"]             # [W, H] — image resolution (new convention)

    # metadata.json stores K at original image resolution.
    # Depth files and cotracker tracks are at depth-model resolution.
    # Scale K down to match depth-map resolution for correct backprojection.
    sample_depth = os.path.join(case_dir, "depth", "0", "0.npy")
    if os.path.exists(sample_depth):
        dH, dW = np.load(sample_depth).shape[:2]
        img_W, img_H = int(WH[0]), int(WH[1])
        if (dW, dH) != (img_W, img_H):
            sx, sy = dW / img_W, dH / img_H
            for K in Ks:
                K[0, 0] *= sx;  K[0, 2] *= sx  # fx, cx
                K[1, 1] *= sy;  K[1, 2] *= sy  # fy, cy
            WH = [dW, dH]

    return c2ws, Ks, WH


def load_tracks(case_dir, cam):
    for subdir in ("cotracker", "track"):
        p = os.path.join(case_dir, subdir, f"{cam}.npz")
        if os.path.exists(p):
            d = np.load(p)
            return d["tracks"], d["visibility"]   # (F,N,2) yx, (F,N) bool
    raise FileNotFoundError(f"No tracks found for cam {cam} in {case_dir}")


def load_depth(case_dir, cam, frame):
    p = os.path.join(case_dir, "depth", str(cam), f"{frame}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def num_cams(case_dir):
    with open(os.path.join(case_dir, "metadata.json")) as f:
        meta = json.load(f)
    return len(meta["K"])


# ── lifting ───────────────────────────────────────────────────────────────────

def lift_tracks_to_world(tracks, visibility, depth, K, c2w, max_points=500):
    """
    Sample up to max_points visible tracks, lift to world space.
    Returns (M, 3) world positions.
    """
    H, W = depth.shape
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    vis_idx = np.where(visibility)[0]
    if len(vis_idx) == 0:
        return np.zeros((0, 3))
    rng = np.random.default_rng(0)
    if len(vis_idx) > max_points:
        vis_idx = rng.choice(vis_idx, max_points, replace=False)

    pts = []
    for idx in vis_idx:
        y, x = int(round(tracks[idx, 0])), int(round(tracks[idx, 1]))
        if not (0 <= y < H and 0 <= x < W):
            continue
        d = depth[y, x]
        if d <= 0:
            continue
        cam_pt = np.array([(x - cx) * d / fx, (y - cy) * d / fy, d, 1.0])
        world_pt = c2w @ cam_pt
        pts.append(world_pt[:3])

    return np.array(pts) if pts else np.zeros((0, 3))


# ── plot ──────────────────────────────────────────────────────────────────────

def make_scatter_plot(case_dir, c2ws, Ks, n_cams, frame=0, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Per-camera world-space track positions  (frame {frame})", fontsize=13)

    for cam in range(n_cams):
        try:
            tracks, vis = load_tracks(case_dir, cam)
            depth = load_depth(case_dir, cam, frame)
        except FileNotFoundError:
            continue
        if depth is None or depth.shape[0] < 2:
            continue

        pts = lift_tracks_to_world(tracks[frame], vis[frame], depth, Ks[cam], c2ws[cam])
        if len(pts) == 0:
            continue

        color = CAM_COLORS[cam % len(CAM_COLORS)]
        axes[0].scatter(pts[:, 0], pts[:, 2], s=4, alpha=0.5, color=color, label=f"cam{cam}")
        axes[1].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, color=color, label=f"cam{cam}")

    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Z (m)"); axes[0].set_title("XZ plane (top view)")
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)"); axes[1].set_title("XY plane (front view)")
    for ax in axes:
        ax.legend(markerscale=3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"Saved plot → {out_path}")
    plt.close()


# ── video ─────────────────────────────────────────────────────────────────────

def _render_frame(c2ws, Ks, n_cams, case_dir, frame, xlim, ylim, zlim, img_w=900, img_h=450):
    fig, axes = plt.subplots(1, 2, figsize=(img_w / 100, img_h / 100), dpi=100)
    fig.suptitle(f"frame {frame:03d}", fontsize=11)

    for cam in range(n_cams):
        try:
            tracks, vis = load_tracks(case_dir, cam)
            depth = load_depth(case_dir, cam, frame)
        except FileNotFoundError:
            continue
        if depth is None or depth.shape[0] < 2:
            continue

        pts = lift_tracks_to_world(tracks[frame], vis[frame], depth, Ks[cam], c2ws[cam])
        if len(pts) == 0:
            continue

        color = CAM_COLORS[cam % len(CAM_COLORS)]
        axes[0].scatter(pts[:, 0], pts[:, 2], s=3, alpha=0.5, color=color, label=f"cam{cam}")
        axes[1].scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, color=color, label=f"cam{cam}")

    for ax, (xl, yl) in zip(axes, [(xlim, zlim), (xlim, ylim)]):
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Z"); axes[0].set_title("XZ (top)")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y"); axes[1].set_title("XY (front)")
    axes[0].legend(markerscale=3, fontsize=7)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def make_video(case_dir, c2ws, Ks, n_cams, out_path, fps=10):
    # First pass: collect axis limits from frame 0 for consistency
    all_pts = {cam: [] for cam in range(n_cams)}
    print("Collecting axis limits from all frames (sampled)...")
    tracks_vis = {}
    for cam in range(n_cams):
        try:
            t, v = load_tracks(case_dir, cam)
            tracks_vis[cam] = (t, v)
        except FileNotFoundError:
            pass

    n_frames = min(t.shape[0] for t, _ in tracks_vis.values()) if tracks_vis else 0
    sample_frames = list(range(0, n_frames, max(1, n_frames // 10)))
    for f in sample_frames:
        for cam, (t, v) in tracks_vis.items():
            depth = load_depth(case_dir, cam, f)
            if depth is None or depth.shape[0] < 2:
                continue
            pts = lift_tracks_to_world(t[f], v[f], depth, Ks[cam], c2ws[cam])
            if len(pts):
                all_pts[cam].append(pts)

    combined = np.concatenate([np.concatenate(v) for v in all_pts.values() if v])
    pad = 0.05
    xlim = (combined[:, 0].min() - pad, combined[:, 0].max() + pad)
    ylim = (combined[:, 1].min() - pad, combined[:, 1].max() + pad)
    zlim = (combined[:, 2].min() - pad, combined[:, 2].max() + pad)

    # Second pass: render video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    print(f"Rendering {n_frames} frames...")
    for f in tqdm(range(n_frames)):
        img = _render_frame(c2ws, Ks, n_cams, case_dir, f, xlim, ylim, zlim)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(img)
    if writer:
        writer.release()
    print(f"Saved video → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--frame", type=int, default=0, help="Frame for static plot")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(args.case_dir, "calibration_analysis")
    os.makedirs(out_dir, exist_ok=True)

    c2ws, Ks, WH = load_calibration(args.case_dir)
    n_cams = len(Ks)
    print(f"Loaded {n_cams} cameras")

    # Static scatter plot
    make_scatter_plot(
        args.case_dir, c2ws, Ks, n_cams,
        frame=args.frame,
        out_path=os.path.join(out_dir, "per_camera_world_positions.png"),
    )

    # Video
    if not args.no_video:
        make_video(
            args.case_dir, c2ws, Ks, n_cams,
            out_path=os.path.join(out_dir, "per_camera_world_positions.mp4"),
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
