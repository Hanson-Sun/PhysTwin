#!/usr/bin/env python3
"""
Visualize first frame depth maps in 3D along with the detected environment plane.
Imports functions from existing visualization scripts.
"""

import os
import json
import numpy as np
import open3d as o3d
from argparse import ArgumentParser

from visualize_3d_scene import create_point_cloud_from_depth, auto_detect_depth_scale, create_camera_frustum, create_camera_axes
from visualize_planes import create_plane_quad


def visualize_depth_with_plane(case_name, depth_root=None, calibrate_path=None, metadata_path=None, 
                               depth_scale=None, auto_scale=True, max_depth=100.0):
    """
    Visualize first frame depth maps and environment plane together.
    
    Args:
        case_name: case name (used to derive default paths)
        depth_root: root directory with depth maps (auto: depth/)
        calibrate_path: path to calibrate.pkl (auto: data/different_types/{case_name}/calibrate.pkl)
        metadata_path: path to metadata.json (auto: data/different_types/{case_name}/metadata.json)
        depth_scale: manual depth scale (auto-detect if None)
        auto_scale: whether to auto-detect depth scale
        max_depth: maximum depth threshold
    """
    # Set default paths from case_name
    if depth_root is None:
        depth_root = f"data/different_types/{case_name}/depth"
    if calibrate_path is None:
        calibrate_path = f"data/different_types/{case_name}/calibrate.pkl"
    if metadata_path is None:
        metadata_path = f"data/different_types/{case_name}/metadata.json"
    # Load calibration
    import pickle
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(calibrate_path, 'rb') as f:
        c2ws_data = pickle.load(f)
    
    cam_indices = list(range(len(c2ws_data)))
    c2ws = [np.array(c2w) for c2w in c2ws_data]
    
    if isinstance(metadata['intrinsics'], dict):
        intrinsics = [np.array(metadata['intrinsics'][str(k)]) for k in cam_indices]
    elif isinstance(metadata['intrinsics'], list):
        intrinsics = [np.array(intr) for intr in metadata['intrinsics']]
    
    stored_wh = metadata.get('WH')
    
    # Find first frame
    first_cam_dir = os.path.join(depth_root, str(cam_indices[0]))
    npy_files = sorted([f for f in os.listdir(first_cam_dir) if f.endswith('.npy')])
    if not npy_files:
        raise RuntimeError(f"No depth maps found in {first_cam_dir}")
    frame_name = os.path.splitext(npy_files[0])[0]
    
    print(f"Visualizing frame: {frame_name}")
    
    geometries = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    # Process depth maps
    for cam_idx, cam_id in enumerate(cam_indices):
        depth_path = os.path.join(depth_root, str(cam_id), f"{frame_name}.npy")
        if not os.path.exists(depth_path):
            print(f"Skipping camera {cam_id} (no depth map)")
            continue
        
        depth = np.load(depth_path).astype(np.float32)
        H, W = depth.shape[:2]
        
        intrinsic = np.array(intrinsics[cam_idx])
        
        # Scale K to depth resolution
        if stored_wh is not None:
            img_W, img_H = int(stored_wh[0]), int(stored_wh[1])
            if (W, H) != (img_W, img_H):
                intrinsic = intrinsic.copy()
                intrinsic[0, 0] *= W / img_W
                intrinsic[0, 2] *= W / img_W
                intrinsic[1, 1] *= H / img_H
                intrinsic[1, 2] *= H / img_H
        
        c2w = c2ws[cam_idx]
        if c2w.shape == (3, 4):
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :4] = c2w
            c2w = c2w_4x4
        
        # Determine depth scale
        if depth_scale is not None:
            ds = depth_scale
        elif auto_scale:
            ds = auto_detect_depth_scale(depth)
        else:
            ds = 1.0
        
        # Create point cloud
        pcd = create_point_cloud_from_depth(depth, intrinsic, c2w, 
                                           depth_scale=ds, max_depth=max_depth)
        pcd.paint_uniform_color(colors[cam_idx % len(colors)])
        geometries.append(pcd)
        
        print(f"Camera {cam_id}: {len(np.asarray(pcd.points))} points")
    
    # Add camera frustums and axes
    for cam_idx, cam_id in enumerate(cam_indices):
        depth_path = os.path.join(depth_root, str(cam_id), f"{frame_name}.npy")
        if not os.path.exists(depth_path):
            continue
        
        depth = np.load(depth_path).astype(np.float32)
        H, W = depth.shape[:2]
        
        intrinsic = np.array(intrinsics[cam_idx])
        
        # Scale K to depth resolution
        if stored_wh is not None:
            img_W, img_H = int(stored_wh[0]), int(stored_wh[1])
            if (W, H) != (img_W, img_H):
                intrinsic = intrinsic.copy()
                intrinsic[0, 0] *= W / img_W
                intrinsic[0, 2] *= W / img_W
                intrinsic[1, 1] *= H / img_H
                intrinsic[1, 2] *= H / img_H
        
        c2w = c2ws[cam_idx]
        if c2w.shape == (3, 4):
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :4] = c2w
            c2w = c2w_4x4
        
        # Create frustum and axes (world space)
        frustum = create_camera_frustum(intrinsic, c2w, W, H, 
                                       frustum_depth=0.5,
                                       color=[1.0, 1.0, 0.0])
        axes = create_camera_axes(c2w, axis_length=0.1)
        
        geometries.append(frustum)
        geometries.append(axes)
        print(f"Camera {cam_id}: frustum and axes added")
    
    # Load and add plane
    plane_path = f"./data/different_types/{case_name}/environment_planes.json"
    if os.path.exists(plane_path):
        with open(plane_path, 'r') as f:
            plane_data = json.load(f)
        
        for i, plane in enumerate(plane_data.get("planes", [])):
            plane_eq = np.array(plane["equation"], dtype=np.float32)
            mesh = create_plane_quad(plane_eq, size=1.0)
            if mesh is not None:
                mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
                geometries.append(mesh)
                print(f"Added plane {i}")
    else:
        print(f"Plane file not found: {plane_path}")
    
    # Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(coord)
    
    print(f"\nVisualizing {len(geometries)} objects...")
    o3d.visualization.draw_geometries(geometries, window_name=f"Depth + Plane - {case_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--case_name", type=str, required=True, help="Case name")
    parser.add_argument("--depth_root", type=str, default=None, help="Root dir with depth maps (default: depth/)")
    parser.add_argument("--calibrate_path", type=str, default=None, help="Path to calibrate.pkl")
    parser.add_argument("--metadata_path", type=str, default=None, help="Path to metadata.json")
    parser.add_argument("--depth_scale", type=float, default=None, help="Manual depth scale")
    parser.add_argument("--auto_scale", action="store_true", default=True)
    parser.add_argument("--max_depth", type=float, default=100.0, help="Max depth threshold")
    args = parser.parse_args()
    
    visualize_depth_with_plane(
        args.case_name,
        depth_root=args.depth_root, 
        calibrate_path=args.calibrate_path, 
        metadata_path=args.metadata_path,
        depth_scale=args.depth_scale, 
        auto_scale=args.auto_scale, 
        max_depth=args.max_depth
    )
