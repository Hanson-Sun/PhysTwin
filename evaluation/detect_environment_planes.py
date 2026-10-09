#!/usr/bin/env python3
"""
Detect environment planes from depth maps using RANSAC.
Supports horizontal (floor/ceiling) and vertical (wall) planes.
"""

import numpy as np
import pickle
import json
import os
from argparse import ArgumentParser
import open3d as o3d


def detect_planes_ransac(points, num_planes=1, distance_threshold=0.01, min_inliers=100):
    """Detect planes using Open3D RANSAC."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    planes = []
    remaining_indices = np.arange(len(points))
    
    for i in range(num_planes):
        if len(pcd.points) < 3:
            break
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        num_inliers = len(inliers)
        if num_inliers < min_inliers:
            break
        
        planes.append((np.array(plane_model, dtype=np.float32), remaining_indices[inliers]))
        
        inlier_mask = np.ones(len(pcd.points), dtype=bool)
        inlier_mask[inliers] = False
        pcd = pcd.select_by_index(np.where(inlier_mask)[0])
        remaining_indices = remaining_indices[inlier_mask]
    
    return planes


def compute_frustum_bounds(intrinsics, c2ws, depth_range=[0.01, 100.0]):
    """Compute AABB of camera frustums."""
    points = []
    
    for intr, c2w in zip(intrinsics, c2ws):
        if isinstance(intr, list):
            intr = np.array(intr)
        if isinstance(c2w, list):
            c2w = np.array(c2w)
        
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        img_h, img_w = int(2*cy), int(2*cx)
        
        min_d, max_d = depth_range
        
        # Near and far plane corners
        for d in [min_d, max_d]:
            for u, v in [(0, 0), (img_w-1, 0), (img_w-1, img_h-1), (0, img_h-1)]:
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                pt = np.array([x, y, d, 1])
                pt_world = c2w @ pt
                points.append(pt_world[:3])
    
    points = np.array(points)
    return points.min(axis=0), points.max(axis=0)


def run_detection(data_path, case_name, output_path, calibrate_path=None, 
                  metadata_path=None, depth_root=None, num_planes=1, 
                  distance_threshold=0.01, min_inliers=100):
    """
    Detect environment planes from depth maps.
    Filters to horizontal (|nz|>0.6) or vertical (|nz|<0.4) planes only.
    """
    # Set defaults
    if calibrate_path is None:
        calibrate_path = f"data/different_types/{case_name}/calibrate.pkl"
    if metadata_path is None:
        metadata_path = f"data/different_types/{case_name}/metadata.json"
    if depth_root is None:
        depth_root = f"data/different_types/{case_name}/depth"
    
    print(f"[DETECT] {case_name} from {depth_root}")
    
    # Load calibration
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(calibrate_path, 'rb') as f:
        c2ws_data = pickle.load(f)
    
    cam_indices = list(range(len(c2ws_data)))
    c2ws = [np.array(c2w) for c2w in c2ws_data]
    
    if isinstance(metadata['intrinsics'], dict):
        intrinsics = [np.array(metadata['intrinsics'][str(k)]) for k in cam_indices]
    else:
        intrinsics = [np.array(intr) for intr in metadata['intrinsics']]
    
    stored_wh = metadata.get('WH')
    
    # Compute frustum bounds
    c2ws_4x4 = []
    for c2w in c2ws:
        if c2w.shape == (3, 4):
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :4] = c2w
            c2ws_4x4.append(c2w_4x4)
        else:
            c2ws_4x4.append(c2w)
    
    min_bound, max_bound = compute_frustum_bounds(intrinsics, c2ws_4x4)
    print(f"[FRUSTUM] Bounds: X=[{min_bound[0]:.2f}, {max_bound[0]:.2f}], Y=[{min_bound[1]:.2f}, {max_bound[1]:.2f}], Z=[{min_bound[2]:.2f}, {max_bound[2]:.2f}]")
    
    # Load depth maps
    all_points = []
    first_cam_dir = os.path.join(depth_root, str(0))
    
    if not os.path.exists(first_cam_dir):
        print(f"[ERROR] Depth directory not found: {first_cam_dir}")
        return {'case_name': case_name, 'planes': []}
    
    npy_files = sorted([f for f in os.listdir(first_cam_dir) if f.endswith('.npy')])
    if not npy_files:
        print(f"[ERROR] No depth maps found")
        return {'case_name': case_name, 'planes': []}
    
    frame_name = os.path.splitext(npy_files[0])[0]
    
    from visualize_3d_scene import create_point_cloud_from_depth, auto_detect_depth_scale
    
    for cam_idx, cam_id in enumerate(cam_indices):
        depth_path = os.path.join(depth_root, str(cam_id), f"{frame_name}.npy")
        if not os.path.exists(depth_path):
            continue
        
        depth = np.load(depth_path).astype(np.float32)
        H, W = depth.shape[:2]
        
        intrinsic = np.array(intrinsics[cam_idx])
        if stored_wh is not None:
            img_W, img_H = int(stored_wh[0]), int(stored_wh[1])
            if (W, H) != (img_W, img_H):
                intrinsic = intrinsic.copy()
                intrinsic[0, 0] *= W / img_W
                intrinsic[0, 2] *= W / img_W
                intrinsic[1, 1] *= H / img_H
                intrinsic[1, 2] *= H / img_H
        
        c2w = c2ws_4x4[cam_idx]
        ds = auto_detect_depth_scale(depth)
        pcd = create_point_cloud_from_depth(depth, intrinsic, c2w, depth_scale=ds, max_depth=100.0)
        
        pts = np.asarray(pcd.points)
        # Filter to frustum
        mask = np.all((pts >= min_bound) & (pts <= max_bound), axis=1)
        pts = pts[mask]
        
        if pts.size > 0:
            all_points.append(pts)
            print(f"[CAM {cam_id}] {len(pts)} points")
    
    if not all_points:
        print("[ERROR] No points in frustum bounds")
        return {'case_name': case_name, 'planes': []}
    
    surface_points = np.vstack(all_points)
    print(f"[TOTAL] {len(surface_points)} points for plane detection")
    
    # Detect planes
    planes_list = detect_planes_ransac(surface_points, num_planes=num_planes,
                                       distance_threshold=distance_threshold,
                                       min_inliers=min_inliers)
    
    # Filter: keep only horizontal or vertical
    result = {'case_name': case_name, 'planes': []}
    
    for plane_eq, inlier_indices in planes_list:
        a, b, c, d = plane_eq
        norm = np.sqrt(a**2 + b**2 + c**2)
        plane_eq = plane_eq / norm
        
        normal_z = abs(plane_eq[2])
        is_horizontal = normal_z > 0.5
        is_vertical = normal_z < 0.5
        
        if not (is_horizontal or is_vertical):
            print(f"[SKIP] |nz|={normal_z:.3f} (angled)")
            continue
        
        plane_type = "horizontal" if is_horizontal else "vertical"
        result['planes'].append({
            'equation': plane_eq.tolist(),
            'normal': plane_eq[:3].tolist(),
            'num_inliers': len(inlier_indices),
            'type': plane_type
        })
        print(f"[{plane_type.upper()}] normal={plane_eq[:3].tolist()}, inliers={len(inlier_indices)}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[SAVED] {len(result['planes'])} plane(s) to {output_path}\n")
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--calibrate_path", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--depth_root", type=str, default=None)
    parser.add_argument("--num_planes", type=int, default=1)
    parser.add_argument("--distance_threshold", type=float, default=0.005)
    parser.add_argument("--min_inliers", type=int, default=100)
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = f"{args.base_path}/{args.case_name}/environment_planes.json"
    
    data_path = f"{args.base_path}/{args.case_name}/final_data.pkl"
    run_detection(data_path, args.case_name, args.output_path,
                  calibrate_path=args.calibrate_path,
                  metadata_path=args.metadata_path,
                  depth_root=args.depth_root,
                  num_planes=args.num_planes,
                  distance_threshold=args.distance_threshold,
                  min_inliers=args.min_inliers)
