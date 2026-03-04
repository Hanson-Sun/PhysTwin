"""
refine_extrinsics_icp.py

Refine DA3-estimated camera extrinsics by ICP-aligning background point clouds
across cameras to cam0 (the reference). Updates calibrate.pkl in-place.

Strategy:
  - Use a static frame (before cloth starts moving) where all cameras see the
    same background (table, floor, walls).
  - Exclude the object/cloth region using the object mask.
  - Backproject each camera's background depth into world space using its
    current DA3 c2w.
  - Run ICP between cam_i background and cam0 background.
  - Apply the corrective transform to cam_i's c2w and save back to calibrate.pkl.

Usage:
  python refine_extrinsics_icp.py --case_dir data/different_types/cloth_double_bend
  python refine_extrinsics_icp.py --case_dir data/different_types/cloth_double_bend --frame 0 --backup
"""

import os
import json
import pickle
import shutil
import argparse
import numpy as np
import open3d as o3d


# ── data loading ──────────────────────────────────────────────────────────────

def load_calibration(case_dir):
    with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)           # list of (4,4) np.ndarray
    with open(os.path.join(case_dir, "metadata.json")) as f:
        meta = json.load(f)
    Ks = [np.asarray(k, dtype=np.float64) for k in meta["intrinsics"]]

    # metadata.json stores K at original image resolution (WH = image dims).
    # depth/ files are at depth-model resolution. Scale K down to match.
    WH = meta.get("WH")
    sample_depth = os.path.join(str(case_dir), "depth", "0", "0.npy")
    if WH and os.path.exists(sample_depth):
        dH, dW = np.load(sample_depth).shape[:2]
        img_W, img_H = int(WH[0]), int(WH[1])
        if (dW, dH) != (img_W, img_H):
            sx, sy = dW / img_W, dH / img_H
            for K in Ks:
                K[0, 0] *= sx;  K[0, 2] *= sx  # fx, cx
                K[1, 1] *= sy;  K[1, 2] *= sy  # fy, cy

    return c2ws, Ks, meta


def save_calibration(case_dir, c2ws):
    path = os.path.join(case_dir, "calibrate.pkl")
    with open(path, "wb") as f:
        pickle.dump(c2ws, f)
    print(f"Saved updated calibration → {path}")


def load_depth(case_dir, cam, frame):
    p = os.path.join(case_dir, "depth", str(cam), f"{frame}.npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def load_object_mask(case_dir, cam, frame):
    """Load the object (cloth) mask for a given camera/frame. Returns bool array or None."""
    mask_pkl = os.path.join(case_dir, "processed_masks.pkl")
    if not os.path.exists(mask_pkl):
        return None
    with open(mask_pkl, "rb") as f:
        masks = pickle.load(f)
    try:
        return masks[frame][cam]["object"].astype(bool)
    except (IndexError, KeyError, TypeError):
        return None


# ── point cloud building ──────────────────────────────────────────────────────

def backproject(depth, K, c2w, mask_exclude=None, max_depth=3.0, max_points=50000):
    """
    Backproject depth map to world-space point cloud.
    mask_exclude: boolean array (H,W) — True = exclude (cloth region)
    Returns open3d.geometry.PointCloud
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = depth.copy()

    # mask out invalid and too-far points
    valid = (d > 0) & (d < max_depth)
    if mask_exclude is not None:
        # resize mask to depth shape if needed
        if mask_exclude.shape != (H, W):
            import cv2
            mask_exclude = cv2.resize(
                mask_exclude.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        valid &= ~mask_exclude

    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    d_v  = d[valid]

    # camera-space coords
    X = (xs_v - cx) * d_v / fx
    Y = (ys_v - cy) * d_v / fy
    Z = d_v
    ones = np.ones_like(Z)
    cam_pts = np.stack([X, Y, Z, ones], axis=1)  # (N, 4)

    # world-space
    world_pts = (c2w @ cam_pts.T).T[:, :3]  # (N, 3)

    # subsample if too many
    if len(world_pts) > max_points:
        idx = np.random.default_rng(0).choice(len(world_pts), max_points, replace=False)
        world_pts = world_pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_pts)
    return pcd


# ── ICP ───────────────────────────────────────────────────────────────────────

def run_icp(source_pcd, target_pcd, voxel_size=0.01, max_iter=200):
    """
    Align source → target via point-to-plane ICP.
    Returns the 4x4 corrective transform T such that T @ source ≈ target.
    """
    # Downsample
    src = source_pcd.voxel_down_sample(voxel_size)
    tgt = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals (needed for point-to-plane)
    radius = voxel_size * 3
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    src.estimate_normals(search)
    tgt.estimate_normals(search)

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * 5,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return result.transformation, result.fitness, result.inlier_rmse


def refine(c2ws, Ks, case_dir, frame=0, max_depth=3.0, voxel_size=0.01):
    """
    Refine camera extrinsics by ICP-aligning each camera's background point
    cloud to camera 0's point cloud.

    Parameters
    ----------
    c2ws      : list of (4, 4) np.ndarray  — cam-to-world poses (input)
    Ks        : list of (3, 3) np.ndarray  — intrinsics
    case_dir  : str or Path                — dataset root (for depth + masks)
    frame     : int                        — frame index to use for ICP
    max_depth : float                      — max depth to include (metres)
    voxel_size: float                      — ICP voxel size (metres)

    Returns
    -------
    new_c2ws : list of (4, 4) np.ndarray  — refined cam-to-world poses
    """
    case_dir = str(case_dir)
    n_cams   = len(c2ws)
    print(f"Refining {n_cams} cameras via ICP using frame {frame}.")

    depth0 = load_depth(case_dir, 0, frame)
    if depth0 is None:
        raise FileNotFoundError(f"Depth not found for cam0 frame {frame}")
    mask0 = load_object_mask(case_dir, 0, frame)
    c2w0  = np.asarray(c2ws[0], dtype=np.float64)
    pcd0  = backproject(depth0, Ks[0], c2w0, mask_exclude=mask0, max_depth=max_depth)
    print(f"  cam0 reference: {len(pcd0.points)} background points")

    new_c2ws = [c2ws[0]]  # cam0 stays fixed

    for cam in range(1, n_cams):
        depth_i = load_depth(case_dir, cam, frame)
        if depth_i is None:
            print(f"  cam{cam}: depth not found — skipping")
            new_c2ws.append(c2ws[cam])
            continue

        mask_i = load_object_mask(case_dir, cam, frame)
        c2w_i  = np.asarray(c2ws[cam], dtype=np.float64)
        pcd_i  = backproject(depth_i, Ks[cam], c2w_i, mask_exclude=mask_i, max_depth=max_depth)
        print(f"  cam{cam}: {len(pcd_i.points)} background points", end="  →  ICP... ", flush=True)

        T, fitness, rmse = run_icp(pcd_i, pcd0, voxel_size=voxel_size)
        new_c2w_i = T @ c2w_i
        new_c2ws.append(new_c2w_i)

        trans  = np.linalg.norm(T[:3, 3])
        angle  = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"fitness={fitness:.3f}  rmse={rmse*100:.1f}cm  correction: t={trans*100:.1f}cm  R={angle:.2f}°")

    return new_c2ws

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--frame", type=int, default=0, help="Frame to use as reference for ICP")
    parser.add_argument("--max_depth", type=float, default=3.0, help="Max depth to include (metres)")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for downsampling (metres)")
    parser.add_argument("--backup", action="store_true", help="Backup calibrate.pkl before overwriting")
    args = parser.parse_args()

    case_dir = args.case_dir
    frame = args.frame

    if args.backup:
        src = os.path.join(case_dir, "calibrate.pkl")
        dst = os.path.join(case_dir, "calibrate_before_icp.pkl")
        shutil.copy2(src, dst)
        print(f"Backed up → {dst}")

    c2ws, Ks, meta = load_calibration(case_dir)
    print(f"Loaded {len(c2ws)} cameras.")

    new_c2ws = refine(c2ws, Ks, case_dir, frame=frame,
                      max_depth=args.max_depth, voxel_size=args.voxel_size)

    save_calibration(case_dir, new_c2ws)
    print("\nDone. Re-run data_process_pcd.py and data_process_track.py to rebuild object_points.")


if __name__ == "__main__":
    main()
