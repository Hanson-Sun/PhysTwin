#!/usr/bin/env python3
"""
Align depth scales across multiple cameras for VDA-inferred depth.

Problem: Video-Depth-Anything runs independently on each camera, producing depth
with arbitrary scale. When lifting to 3D, the point clouds won't be consistent.

Solution: Find per-camera scale factors that maximize 3D point cloud alignment.

Required inputs:
- Camera calibration: metadata.json (intrinsics) + calibrate.pkl (extrinsics)
- VDA depth: Output from infer_depth.py (data/vda_depth/{case}/)

NO baseline/reference depth is required - alignment uses cross-view consistency.

Methods:
1. Cross-Projection Consistency (default): Project camera i's points into 
   camera j's view and compare depths. Optimize scales to minimize disagreement.

2. Chamfer Distance: Generate 3D point clouds and minimize Chamfer distance
   between overlapping regions.

Usage:
    python align_depth.py --case_name double_stretch_sloth
    python align_depth.py --case_name double_stretch_sloth --output_dir data/vda_depth_aligned
    python align_depth.py  # Process all cases in data_config.csv
"""

import argparse
import numpy as np
import os
import cv2
import pickle
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import warnings
from tqdm import tqdm

# For video rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def normalize_depth_for_vis(depth, vmin=None, vmax=None):
    """Normalize depth for visualization (copied from compare_depth.py)"""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if len(valid) == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    if vmin is None:
        vmin = np.percentile(valid, 2)
    if vmax is None:
        vmax = np.percentile(valid, 98)
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    return depth_vis

def render_depth_video(depth_dir_before, depth_dir_after, output_dir, camera_id, max_frames=-1):
    """Render before/after depth video for a camera"""
    depth_dir_before = Path(depth_dir_before)
    depth_dir_after = Path(depth_dir_after)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load frames
    npy_files = sorted(depth_dir_before.glob('*.npy'), key=lambda x: int(x.stem))
    if max_frames > 0:
        npy_files = npy_files[:max_frames]
    frames = []
    for f in npy_files:
        name = f.stem
        depth_before = np.load(f)
        after_path = depth_dir_after / f.name
        if not after_path.exists():
            continue
        depth_after = np.load(after_path)
        # Visualize
        vmin = np.percentile(depth_before[np.isfinite(depth_before) & (depth_before > 0)], 2) if np.any((depth_before > 0) & np.isfinite(depth_before)) else 0
        vmax = np.percentile(depth_before[np.isfinite(depth_before) & (depth_before > 0)], 98) if np.any((depth_before > 0) & np.isfinite(depth_before)) else 1
        vis_before = normalize_depth_for_vis(depth_before, vmin, vmax)
        vis_after = normalize_depth_for_vis(depth_after, vmin, vmax)
        # Stack side by side
        frame = np.hstack([vis_before, vis_after])
        # Add text
        cv2.putText(frame, 'Before', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'After', (frame.shape[1]//2 + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Frame {name}', (30, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        frames.append(frame)
    if not frames:
        print(f"  No frames found for camera {camera_id} in {depth_dir_before}")
        return
    h, w = frames[0].shape[:2]
    video_path = output_dir / f'before_after_camera_{camera_id}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  Saved before/after video: {video_path}")
warnings.filterwarnings('ignore')


def load_calibration(case_dir):
    """Load camera intrinsics and extrinsics"""
    case_dir = Path(case_dir)
    
    # Load intrinsics from metadata.json
    metadata_path = case_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {case_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    intrinsics = [np.array(K) for K in metadata['intrinsics']]
    
    # Load extrinsics (camera-to-world) from calibrate.pkl
    calibrate_path = case_dir / 'calibrate.pkl'
    if not calibrate_path.exists():
        raise FileNotFoundError(f"calibrate.pkl not found in {case_dir}")
    
    with open(calibrate_path, 'rb') as f:
        extrinsics = pickle.load(f)  # List of 4x4 c2w matrices
    
    return intrinsics, extrinsics, metadata


def load_depth_frames(depth_dir, max_frames=-1):
    """Load depth frames from a directory"""
    depth_dir = Path(depth_dir)
    if not depth_dir.exists():
        return {}, []
    
    npy_files = sorted(depth_dir.glob('*.npy'), key=lambda x: int(x.stem))
    if max_frames > 0:
        npy_files = npy_files[:max_frames]
    
    depths = {}
    names = []
    for f in npy_files:
        depths[f.stem] = np.load(f)
        names.append(f.stem)
    
    return depths, names


# Note: CoTracker data is NOT used for alignment because:
# 1. CoTracker runs independently per camera (no cross-camera correspondences)
# 2. We don't have reference depth to create 3D ground truth tracks
# Instead, we use cross-projection consistency which only needs calibration + VDA depth


def unproject_points(depths, intrinsic, extrinsic, scale=1.0, subsample=10):
    """
    Unproject depth map to 3D point cloud in world coordinates.
    
    Args:
        depths: HxW depth map
        intrinsic: 3x3 camera intrinsic matrix
        extrinsic: 4x4 camera-to-world matrix
        scale: depth scale factor
        subsample: subsample factor to reduce points
    
    Returns:
        Nx3 point cloud in world coordinates
    """
    h, w = depths.shape
    
    # Create pixel grid
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    
    # Get depths at these pixels
    d = depths[v, u] * scale
    
    # Filter invalid depths
    valid = (d > 0) & np.isfinite(d)
    u, v, d = u[valid], v[valid], d[valid]
    
    if len(d) == 0:
        return np.zeros((0, 3))
    
    # Unproject to camera coordinates
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    
    # Stack as Nx4 homogeneous coordinates
    points_cam = np.stack([x, y, z, np.ones_like(x)], axis=1)
    
    # Transform to world coordinates
    points_world = (extrinsic @ points_cam.T).T[:, :3]
    
    return points_world


def unproject_pixel(u, v, depth, intrinsic, extrinsic, scale=1.0):
    """Unproject a single pixel to 3D world coordinates"""
    d = depth * scale
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    
    point_cam = np.array([x, y, z, 1.0])
    point_world = (extrinsic @ point_cam)[:3]
    
    return point_world


# Note: Track-based alignment is not implemented because CoTracker doesn't provide
# cross-camera correspondences. We use cross-projection consistency instead.


def compute_cross_projection_loss(scales, depths_all, intrinsics, extrinsics, frame_idx=0, n_samples=1000):
    """
    Compute loss based on cross-projection depth consistency.
    
    Project points from camera i into camera j's view and compare depths.
    """
    scales = np.array(scales)
    
    total_loss = 0.0
    n_comparisons = 0
    
    # Compare each pair of cameras
    camera_pairs = [(0, 1), (0, 2), (1, 2)]
    
    for cam_i, cam_j in camera_pairs:
        if cam_i not in depths_all or cam_j not in depths_all:
            continue
        
        depth_i = depths_all[cam_i]
        depth_j = depths_all[cam_j]
        
        K_i, K_j = intrinsics[cam_i], intrinsics[cam_j]
        T_i, T_j = extrinsics[cam_i], extrinsics[cam_j]
        s_i, s_j = scales[cam_i], scales[cam_j]
        
        h, w = depth_i.shape
        
        # Sample random pixels from camera i
        valid_mask = (depth_i > 0) & np.isfinite(depth_i)
        valid_coords = np.where(valid_mask)
        
        if len(valid_coords[0]) < n_samples:
            continue
        
        # Random sample
        indices = np.random.choice(len(valid_coords[0]), min(n_samples, len(valid_coords[0])), replace=False)
        v_samples = valid_coords[0][indices]
        u_samples = valid_coords[1][indices]
        d_samples = depth_i[v_samples, u_samples] * s_i
        
        # Unproject to 3D (world coordinates)
        fx_i, fy_i = K_i[0, 0], K_i[1, 1]
        cx_i, cy_i = K_i[0, 2], K_i[1, 2]
        
        x = (u_samples - cx_i) * d_samples / fx_i
        y = (v_samples - cy_i) * d_samples / fy_i
        z = d_samples
        
        points_cam_i = np.stack([x, y, z, np.ones_like(x)], axis=1)
        points_world = (T_i @ points_cam_i.T).T[:, :3]
        
        # Project into camera j
        T_j_inv = np.linalg.inv(T_j)
        points_cam_j = (T_j_inv @ np.hstack([points_world, np.ones((len(points_world), 1))]).T).T[:, :3]
        
        # Only keep points in front of camera j
        in_front = points_cam_j[:, 2] > 0.01
        if not np.any(in_front):
            continue
        
        points_cam_j = points_cam_j[in_front]
        
        # Project to image plane
        fx_j, fy_j = K_j[0, 0], K_j[1, 1]
        cx_j, cy_j = K_j[0, 2], K_j[1, 2]
        
        u_j = (points_cam_j[:, 0] * fx_j / points_cam_j[:, 2] + cx_j).astype(int)
        v_j = (points_cam_j[:, 1] * fy_j / points_cam_j[:, 2] + cy_j).astype(int)
        
        # Check bounds
        h_j, w_j = depth_j.shape
        in_bounds = (u_j >= 0) & (u_j < w_j) & (v_j >= 0) & (v_j < h_j)
        
        if not np.any(in_bounds):
            continue
        
        u_j = u_j[in_bounds]
        v_j = v_j[in_bounds]
        expected_depth = points_cam_j[in_bounds, 2]  # Depth in camera j coordinates
        
        # Get actual depth from camera j (scaled)
        actual_depth = depth_j[v_j, u_j] * s_j
        
        # Only compare valid depths
        valid = (actual_depth > 0) & np.isfinite(actual_depth)
        if not np.any(valid):
            continue
        
        expected_depth = expected_depth[valid]
        actual_depth = actual_depth[valid]
        
        # Compute relative depth error (scale-invariant)
        # Use log-space for better optimization landscape
        log_diff = np.log(expected_depth + 1e-6) - np.log(actual_depth + 1e-6)
        loss = np.mean(log_diff ** 2)
        
        total_loss += loss
        n_comparisons += 1
    
    if n_comparisons == 0:
        return 1e6
    
    return total_loss / n_comparisons


def compute_chamfer_distance(pc1, pc2, max_points=5000):
    """Compute Chamfer distance between two point clouds"""
    if len(pc1) == 0 or len(pc2) == 0:
        return 1e6
    
    # Subsample if too many points
    if len(pc1) > max_points:
        idx = np.random.choice(len(pc1), max_points, replace=False)
        pc1 = pc1[idx]
    if len(pc2) > max_points:
        idx = np.random.choice(len(pc2), max_points, replace=False)
        pc2 = pc2[idx]
    
    # Build KD-trees
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    # Query nearest neighbors
    dist1, _ = tree2.query(pc1, k=1)
    dist2, _ = tree1.query(pc2, k=1)
    
    # Chamfer distance
    chamfer = np.mean(dist1) + np.mean(dist2)
    
    return chamfer


def compute_point_cloud_loss(scales, depths_all, intrinsics, extrinsics, frame_idx=0):
    """
    Compute loss based on point cloud overlap/Chamfer distance.
    """
    scales = np.array([1.0, scales[0], scales[1]])  # Fix camera 0 scale to 1
    
    # Generate point clouds for each camera
    point_clouds = []
    for cam_id in range(3):
        if cam_id not in depths_all:
            point_clouds.append(np.zeros((0, 3)))
            continue
        
        pc = unproject_points(
            depths_all[cam_id], 
            intrinsics[cam_id], 
            extrinsics[cam_id],
            scale=scales[cam_id],
            subsample=20
        )
        point_clouds.append(pc)
    
    # Compute pairwise Chamfer distances
    total_loss = 0.0
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for i, j in pairs:
        if len(point_clouds[i]) > 0 and len(point_clouds[j]) > 0:
            chamfer = compute_chamfer_distance(point_clouds[i], point_clouds[j])
            total_loss += chamfer
    
    return total_loss


def optimize_scales(depths_all, intrinsics, extrinsics,
                    method='cross_projection', n_frames=-1, verbose=True):
    """
    Optimize scale factors for each camera using cross-view consistency.
    
    No reference depth is needed - uses geometric constraints from calibration.
    
    Args:
        depths_all: Dict of {cam_id: {frame_name: depth_array}}
        intrinsics: List of 3x3 intrinsic matrices
        extrinsics: List of 4x4 extrinsic matrices (camera-to-world)
        method: 'cross_projection' or 'chamfer'
        n_frames: Number of frames to use for optimization
        verbose: Print progress
    
    Returns:
        scales: [s0, s1, s2] where s0 = 1.0 (camera 0 is reference)
    """
    # Get common frame names across all cameras
    frame_names = None
    for cam_id in depths_all:
        names = set(depths_all[cam_id].keys())
        if frame_names is None:
            frame_names = names
        else:
            frame_names = frame_names & names
    
    frame_names = sorted(frame_names, key=int)
    
    if len(frame_names) == 0:
        print("  Warning: No common frames found across cameras")
        return [1.0, 1.0, 1.0]
    
    # Select frames for optimization
    if n_frames is not None and n_frames > 0 and len(frame_names) > n_frames:
        indices = np.linspace(0, len(frame_names) - 1, n_frames, dtype=int)
        selected_frames = [frame_names[i] for i in indices]
    else:
        selected_frames = frame_names
    if verbose:
        print(f"  Optimizing scales using {len(selected_frames)} frames...")
        print(f"  Method: {method}")
    
    # --- Caching: Precompute per-frame depths dicts ---
    frame_depths_list = []
    for frame_name in selected_frames:
        frame_depths_list.append({cam_id: depths_all[cam_id][frame_name] for cam_id in depths_all})

    # --- Parallelism: Use joblib for per-frame loss ---
    try:
        from joblib import Parallel, delayed
        n_jobs = min(8, os.cpu_count() or 1)
    except ImportError:
        Parallel = None
        n_jobs = 1

    def per_frame_loss(scales, frame_depths):
        if method == 'cross_projection':
            return compute_cross_projection_loss(scales, frame_depths, intrinsics, extrinsics)
        elif method == 'chamfer':
            return compute_point_cloud_loss(scales, frame_depths, intrinsics, extrinsics)
        else:
            return compute_cross_projection_loss(scales, frame_depths, intrinsics, extrinsics)

    def loss_fn(scales):
        # Vectorized/parallel per-frame loss
        if Parallel is not None and n_jobs > 1:
            losses = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(per_frame_loss)(scales, frame_depths)
                for frame_depths in frame_depths_list
            )
            total_loss = np.sum(losses)
        else:
            total_loss = 0.0
            for frame_depths in frame_depths_list:
                total_loss += per_frame_loss(scales, frame_depths)
        return total_loss / len(frame_depths_list)
    
    # Initial guess: all scales = 1
    n_cams = len(depths_all)
    x0 = np.ones(n_cams)
    bounds = [(0.1, 10.0)] * n_cams
    
    # Optimize
    if verbose:
        print(f"  Initial loss: {loss_fn(x0):.6f}")
    
    # Callback for printing progress
    history = {'iter': 0}
    def callback(xk):
        history['iter'] += 1
        current_loss = loss_fn(xk)
        print(f"  Iter {history['iter']:3d}: loss={current_loss:.6f} scales=[1.0, {xk[0]:.4f}, {xk[1]:.4f}]")

    result = minimize(
        loss_fn,
        x0,
        # method='L-BFGS-B',
        method='Powell',
        bounds=bounds,
        options={'maxiter': 500, 'disp': verbose},
        callback=callback if verbose else None
    )

    # Normalize scales so their mean is 1 (no camera privileged)
    scales = result.x
    scales = scales / np.mean(scales)
    optimal_scales = scales.tolist()

    if verbose:
        print(f"  Final loss: {result.fun:.6f}")
        print(f"  Optimal scales (mean=1): {optimal_scales}")

    return optimal_scales


def apply_scales_and_save(input_dir, output_dir, scales, case_name):
    """Apply scale factors and save aligned depth"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    for cam_id in range(3):
        cam_input = input_dir / str(cam_id)
        cam_output = output_dir / str(cam_id)
        
        if not cam_input.exists():
            continue
        
        cam_output.mkdir(parents=True, exist_ok=True)
        
        scale = scales[cam_id]
        
        # Process each depth file
        npy_files = sorted(cam_input.glob('*.npy'), key=lambda x: int(x.stem))
        
        for f in npy_files:
            depth = np.load(f)
            depth_scaled = depth * scale
            np.save(cam_output / f.name, depth_scaled)
        
        # Copy visualization video if exists
        vis_video = cam_input / 'depth_vis.mp4'
        if vis_video.exists():
            import shutil
            shutil.copy(vis_video, cam_output / 'depth_vis.mp4')
    
    # Save scale factors
    scale_file = output_dir / 'scales.json'
    with open(scale_file, 'w') as f:
        json.dump({
            'case_name': case_name,
            'scales': scales,
            'description': 'Scale factors applied to each camera (camera 0 is reference)'
        }, f, indent=2)
    
    print(f"  Saved aligned depth to {output_dir}")
    print(f"  Scale factors: {scales}")


def align_case(case_name, base_path='./data/different_types', 
               vda_path='./data/vda_depth', output_path=None,
               method='cross_projection', max_frames=-1):
    """
    Align depth scales for a single case.
    
    Args:
        case_name: Name of the case
        base_path: Path to case data (for calibration)
        vda_path: Path to VDA depth output
        output_path: Output path for aligned depth (default: vda_path + '_aligned')
        method: Alignment method
        max_frames: Max frames to load (-1 for all)
    """
    print(f"\nProcessing case: {case_name}")
    
    case_dir = Path(base_path) / case_name
    vda_case_dir = Path(vda_path) / case_name
    
    if output_path is None:
        output_dir = Path(vda_path + '_aligned') / case_name
    else:
        output_dir = Path(output_path) / case_name
    
    # Check if VDA depth exists
    if not vda_case_dir.exists():
        print(f"  Error: VDA depth not found at {vda_case_dir}")
        return None
    
    # Load calibration
    try:
        intrinsics, extrinsics, metadata = load_calibration(case_dir)
        print(f"  Loaded calibration for {len(intrinsics)} cameras")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return None
    
    # Load depth for all cameras
    depths_all = {}
    for cam_id in range(3):
        cam_depth_dir = vda_case_dir / str(cam_id)
        depths, names = load_depth_frames(cam_depth_dir, max_frames)
        if len(depths) > 0:
            depths_all[cam_id] = depths
            print(f"  Camera {cam_id}: loaded {len(depths)} frames")
    
    if len(depths_all) < 2:
        print(f"  Error: Need at least 2 cameras for alignment, found {len(depths_all)}")
        return None
    
    # Optimize scales using cross-view consistency (no reference depth needed)
    scales = optimize_scales(
        depths_all, intrinsics, extrinsics,
        method=method,
        n_frames=max_frames,
        verbose=True
    )
    
    # Apply scales and save
    apply_scales_and_save(vda_case_dir, output_dir, scales, case_name)
    
    return scales


def main():
    parser = argparse.ArgumentParser(description='Align depth scales across cameras')
    parser.add_argument('--case_name', type=str, default=None,
                        help='Specific case to process (default: all from data_config.csv)')
    parser.add_argument('--base_path', type=str, default='./data/different_types',
                        help='Base path to case data')
    parser.add_argument('--vda_path', type=str, default='./data/vda_depth',
                        help='Path to VDA depth output')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: vda_path + "_aligned")')
    parser.add_argument('--method', type=str, default='cross_projection',
                        choices=['cross_projection', 'chamfer'],
                        help='Alignment method')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Max frames to load per camera (-1 for all)')
    parser.add_argument('--n_frames', type=int, default=-1,
                        help='Number of frames to use for scale optimization (-1 for all)')
    parser.add_argument('--render_video', action='store_true',
                        help='Render before/after depth video for each camera')
    args = parser.parse_args()

    if args.case_name:
        # Process single case
        scales = align_case(
            args.case_name,
            base_path=args.base_path,
            vda_path=args.vda_path,
            output_path=args.output_dir,
            method=args.method,
            max_frames=args.max_frames,
            n_frames=args.n_frames
        )
        if args.render_video:
            # Render before/after video for each camera
            case = args.case_name
            vda_dir = Path(args.vda_path) / case
            vda_aligned_dir = Path(args.output_dir) / case if args.output_dir else Path(args.vda_path + '_aligned') / case
            out_dir = vda_aligned_dir / 'videos'
            for cam_id in range(3):
                render_depth_video(vda_dir / str(cam_id), vda_aligned_dir / str(cam_id), out_dir, cam_id, max_frames=args.max_frames)
    else:
        # Process all cases from data_config.csv
        config_file = 'data_config.csv'
        if not os.path.exists(config_file):
            print(f"Error: {config_file} not found")
            return
        with open(config_file, 'r') as f:
            lines = f.readlines()
        cases = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 1:
                    cases.append(parts[0])
        print(f"Found {len(cases)} cases in {config_file}")
        all_scales = {}
        for case_name in cases:
            scales = align_case(
                case_name,
                base_path=args.base_path,
                vda_path=args.vda_path,
                output_path=args.output_dir,
                method=args.method,
                max_frames=args.max_frames
            )
            if scales:
                all_scales[case_name] = scales
            if args.render_video:
                vda_dir = Path(args.vda_path) / case_name
                vda_aligned_dir = Path(args.output_dir) / case_name if args.output_dir else Path(args.vda_path + '_aligned') / case_name
                out_dir = vda_aligned_dir / 'videos'
                for cam_id in range(3):
                    render_depth_video(vda_dir / str(cam_id), vda_aligned_dir / str(cam_id), out_dir, cam_id, max_frames=args.max_frames)
        # Summary
        print("\n" + "="*50)
        print("ALIGNMENT SUMMARY")
        print("="*50)
        for case_name, scales in all_scales.items():
            print(f"  {case_name}: {scales}")


if __name__ == '__main__':
    main()
