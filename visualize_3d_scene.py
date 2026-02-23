#!/usr/bin/env python3
"""
Visualize depth maps from multiple cameras in 3D using Open3D.
Loads camera intrinsics, c2w transforms, and depth maps to create point clouds.
"""
import os
import argparse
import pickle
import json

import numpy as np
import open3d as o3d


def create_point_cloud_from_depth(depth, intrinsic, c2w, depth_scale=1.0, max_depth=100.0):
    """
    Create a point cloud from a depth map using camera intrinsics and extrinsics.
    
    Args:
        depth: (H, W) depth map array
        intrinsic: (3, 3) camera intrinsic matrix
        c2w: (4, 4) camera-to-world transformation matrix
        depth_scale: scale factor to convert depth values to meters (depth_in_meters = depth_value / depth_scale)
        max_depth: maximum depth threshold in meters
    
    Returns:
        o3d.geometry.PointCloud
    """
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    
    # Filter valid depths
    valid_mask = np.isfinite(z) & (z > 0)
    if depth_scale > 0:
        z_meters = z / depth_scale
        valid_mask &= (z_meters < max_depth)
    else:
        z_meters = z
        valid_mask &= (z < max_depth)
    
    u = u[valid_mask]
    v = v[valid_mask]
    z = z_meters[valid_mask]
    
    # Backproject to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (N, 3) array
    pts_cam = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates
    pts_cam_homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (c2w @ pts_cam_homo.T).T[:, :3]
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    
    return pcd


def create_camera_frustum(intrinsic, c2w, width, height, frustum_depth=1.0, color=[1.0, 1.0, 0.0], line_width=1.0):
    """
    Create a wireframe frustum visualization for a camera.
    
    Args:
        intrinsic: (3, 3) camera intrinsic matrix
        c2w: (4, 4) camera-to-world transformation
        width: image width in pixels
        height: image height in pixels
        frustum_depth: distance from camera center to frustum plane
        color: RGB color for the frustum lines
        line_width: width of the lines
    
    Returns:
        o3d.geometry.LineSet
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Define image corners in pixel coordinates
    corners_px = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float64)
    
    # Camera origin
    origin = np.array([0, 0, 0])
    
    # Backproject corners to camera coordinates at frustum_depth
    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) * frustum_depth / fx
        y = (v - cy) * frustum_depth / fy
        z = frustum_depth
        corners_cam.append([x, y, z])
    
    # Combine origin and corners
    pts_cam = np.vstack([[0, 0, 0], corners_cam])  # (5, 3)
    
    # Transform to world coordinates
    pts_cam_homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (c2w @ pts_cam_homo.T).T[:, :3]
    
    # Define lines: origin to each corner + rectangle edges
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # origin to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # rectangle edges
    ]
    
    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    
    return line_set


def create_camera_axes(c2w, axis_length=0.1):
    """
    Create coordinate axes at the camera position.
    
    Args:
        c2w: (4, 4) camera-to-world transformation
        axis_length: length of each axis
    
    Returns:
        o3d.geometry.TriangleMesh
    """
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
    axes.transform(c2w)
    return axes


def auto_detect_depth_scale(depth):
    """
    Automatically detect the depth scale based on the median depth value.
    Assumes depth values should typically be in the range 0.1-10 meters.
    
    Returns a scale such that depth_in_meters = depth_value / scale
    """
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return 1.0
    
    median_val = np.median(valid)
    
    # Try different scales to get median in reasonable range (0.1 - 10 meters)
    candidates = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    for scale in candidates:
        median_meters = median_val / scale
        if 0.1 <= median_meters <= 10.0:
            return scale
    
    return 1.0


def visualize_depth_scene(depth_root, calibrate_path=None, metadata_path=None,
                          intrinsics=None, extrinsics=None,
                          frame_name=None, 
                          depth_scale=None,
                          auto_scale=True,
                          show_cameras=True,
                          frustum_scale=0.5,
                          axis_scale=0.1,
                          max_depth=100.0):
    """
    Main visualization function.
    
    Args:
        depth_root: root directory containing camera subdirectories with depth maps
        calibrate_path: path to calibrate.pkl containing c2w matrices (optional if intrinsics/extrinsics provided)
        metadata_path: path to metadata.json containing intrinsics (optional if intrinsics/extrinsics provided)
        intrinsics: (num_cams, 3, 3) camera intrinsic matrices (optional, loads from metadata_path if None)
        extrinsics: (num_cams, 4, 4) extrinsic matrices world-to-camera (optional, computed from calibrate_path if None)
        frame_name: specific frame to visualize (without .npy extension)
        depth_scale: manual depth scale factor (if None, auto-detect)
        auto_scale: whether to auto-detect depth scale
        show_cameras: whether to show camera frustums and axes
        frustum_scale: scale factor for frustum depth (relative to image diagonal)
        axis_scale: scale factor for camera axes (relative to image diagonal)
        max_depth: maximum depth threshold in meters
    """
    # Load calibration from files if not provided
    if intrinsics is None or extrinsics is None:
        if metadata_path is None or calibrate_path is None:
            raise ValueError("Must provide either (intrinsics, extrinsics) or (metadata_path, calibrate_path)")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        with open(calibrate_path, 'rb') as f:
            c2ws_data = pickle.load(f)
        
        if isinstance(c2ws_data, list):
            cam_indices = list(range(len(c2ws_data)))
            c2ws = [np.array(c2w) for c2w in c2ws_data]
        else:
            raise ValueError(f"Unexpected calibrate.pkl format: {type(c2ws_data)}")
        
        if isinstance(metadata['intrinsics'], dict):
            intrinsics = [np.array(metadata['intrinsics'][str(k)]) for k in cam_indices]
        elif isinstance(metadata['intrinsics'], list):
            intrinsics = [np.array(intr) for intr in metadata['intrinsics']]
        else:
            raise ValueError(f"Unexpected intrinsics format: {type(metadata['intrinsics'])}")

        # Convert c2w to w2c
        extrinsics = [np.linalg.inv(c2w if c2w.shape == (4,4) else np.vstack([c2w, [0,0,0,1]])) for c2w in c2ws]
    else:
        cam_indices = list(range(len(intrinsics)))
        # Convert extrinsics (w2c) to c2w if needed
        c2ws = [np.linalg.inv(extrinsics[i] if extrinsics[i].shape == (4, 4) else np.vstack([extrinsics[i], [0,0,0,1]])) for i in range(len(extrinsics))]
        intrinsics = [np.array(intr) for intr in intrinsics]
    
    # Auto-select first frame if not specified
    if frame_name is None:
        first_cam_dir = os.path.join(depth_root, str(cam_indices[0]))
        npy_files = sorted([f for f in os.listdir(first_cam_dir) if f.endswith('.npy')])
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {first_cam_dir}")
        frame_name = os.path.splitext(npy_files[0])[0]
    
    print(f"Visualizing frame: {frame_name}")
    print(f"Number of cameras: {len(cam_indices)}")
    
    # Process each camera
    point_clouds = []
    camera_geometries = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    for cam_idx, cam_id in enumerate(cam_indices):
        # Load depth map
        depth_path = os.path.join(depth_root, str(cam_id), f"{frame_name}.npy")
        if not os.path.exists(depth_path):
            print(f"Warning: {depth_path} not found, skipping camera {cam_id}")
            continue
        
        depth = np.load(depth_path).astype(np.float32)
        H, W = depth.shape

        # Get camera parameters
        intrinsic = intrinsics[cam_idx]
        if isinstance(intrinsic, list):
            intrinsic = np.array(intrinsic)

        c2w = c2ws[cam_idx]

        # Ensure c2w is 4x4
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
        
        # Debug output
        print(f"Camera {cam_id}: depth_scale = {ds:.1f}, shape = {depth.shape}")
        print(f"C2W shape: {c2w.shape}")
        print(f"C2W matrix:\n{c2w}")
        print(f"Depth stats: {depth.min()} {depth.max()} {depth.mean()}")
        print(f"Intrinsic:\n{intrinsic}")
        print()
        
        # Create point cloud
        pcd = create_point_cloud_from_depth(depth, intrinsic, c2w, depth_scale=ds, max_depth=max_depth)
        pcd.paint_uniform_color(colors[cam_idx % len(colors)])
        point_clouds.append(pcd)
    
    if not point_clouds:
        raise RuntimeError("No point clouds created. Check depth file paths.")
    
    print(f"Created {len(point_clouds)} point clouds")
    all_points = []
    for pcd in point_clouds:
        pts = np.asarray(pcd.points)
        if pts.size > 0:
            all_points.append(pts)
    
    if all_points:
        all_points = np.vstack(all_points)
        scene_center = np.mean(all_points, axis=0)
        # Compute scene extent (bounding box diagonal)
        scene_min = all_points.min(axis=0)
        scene_max = all_points.max(axis=0)
        scene_extent = np.linalg.norm(scene_max - scene_min)
    else:
        scene_center = np.zeros(3)
        scene_extent = 1.0
    
    # Now create camera visualizations based on scene extent
    if show_cameras:
        for cam_idx, cam_id in enumerate(cam_indices):
            depth_path = os.path.join(depth_root, str(cam_id), f"{frame_name}.npy")
            if not os.path.exists(depth_path):
                continue
            
            depth = np.load(depth_path).astype(np.float32)
            H, W = depth.shape

            intrinsic = intrinsics[cam_idx]
            if isinstance(intrinsic, list):
                intrinsic = np.array(intrinsic)

            c2w = c2ws[cam_idx]
            if c2w.shape == (3, 4):
                c2w_4x4 = np.eye(4)
                c2w_4x4[:3, :4] = c2w
                c2w = c2w_4x4
            
            # Size cameras based on scene extent, not depth scale
            frustum_depth = scene_extent * frustum_scale
            axis_length = scene_extent * axis_scale
            
            frustum = create_camera_frustum(intrinsic, c2w, W, H, 
                                           frustum_depth=frustum_depth,
                                           color=[1.0, 1.0, 0.0],
                                           line_width=1.0)
            axes = create_camera_axes(c2w, axis_length=axis_length)
            
            camera_geometries.append(frustum)
            camera_geometries.append(axes)
    
    print(f"Scene extent: {scene_extent:.3f}")
    print(f"Scene center: {scene_center}")
        
    # Store point cloud info for later camera sizing
    point_clouds.append(pcd)
    
    if not point_clouds:
        raise RuntimeError("No point clouds created. Check depth file paths.")
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'3D Depth Visualization - Frame: {frame_name}')
    
    # Add geometries
    for pcd in point_clouds:
        vis.add_geometry(pcd)
    for geom in camera_geometries:
        vis.add_geometry(geom)
    
    # Set view control to look from first camera
    vis.poll_events()
    vis.update_renderer()
    
    ctr = vis.get_view_control()
    
    # Compute scene center from point clouds
    all_points = []
    for pcd in point_clouds:
        pts = np.asarray(pcd.points)
        if pts.size > 0:
            all_points.append(pts)
    
    if all_points:
        all_points = np.vstack(all_points)
        scene_center = np.mean(all_points, axis=0)
        
        # Set initial viewpoint from first camera
        c2w_0 = c2ws[0]
        if c2w_0.shape == (3, 4):
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :4] = c2w_0
            c2w_0 = c2w_4x4
        
        cam_pos = c2w_0[:3, 3]
        cam_forward = c2w_0[:3, :3] @ np.array([0, 0, 1])
        cam_up = c2w_0[:3, :3] @ np.array([0, -1, 0])
        
        front = scene_center - cam_pos
        front = front / (np.linalg.norm(front) + 1e-8)
        up = cam_up / (np.linalg.norm(cam_up) + 1e-8)
        
        ctr.set_lookat(scene_center.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up(up.tolist())
        ctr.set_zoom(0.5)
    
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-camera depth maps in 3D')
    parser.add_argument('--depth_root', required=True, 
                       help='Root directory containing camera subdirectories with depth maps')
    parser.add_argument('--calibrate', required=True,
                       help='Path to calibrate.pkl file containing c2w transforms')
    parser.add_argument('--metadata', required=True,
                       help='Path to metadata.json file containing camera intrinsics')
    parser.add_argument('--frame', default=None,
                       help='Frame name to visualize (without .npy extension). If not specified, uses first frame.')
    parser.add_argument('--depth_scale', type=float, default=None,
                       help='Manual depth scale factor (depth_in_meters = depth_value / depth_scale)')
    parser.add_argument('--no_auto_scale', action='store_true',
                       help='Disable automatic depth scale detection')
    parser.add_argument('--no_cameras', action='store_true',
                       help='Do not show camera frustums and axes')
    parser.add_argument('--frustum_scale', type=float, default=0.05,
                       help='Scale factor for frustum depth (relative to image diagonal)')
    parser.add_argument('--axis_scale', type=float, default=0.01,
                       help='Scale factor for camera axes (relative to image diagonal)')
    parser.add_argument('--max_depth', type=float, default=100.0,
                       help='Maximum depth threshold in meters')
    
    args = parser.parse_args()
    
    visualize_depth_scene(
        depth_root=args.depth_root,
        calibrate_path=args.calibrate,
        metadata_path=args.metadata,
        frame_name=args.frame,
        depth_scale=args.depth_scale,
        auto_scale=not args.no_auto_scale,
        show_cameras=not args.no_cameras,
        frustum_scale=args.frustum_scale,
        axis_scale=args.axis_scale,
        max_depth=args.max_depth
    )


if __name__ == '__main__':
    main()