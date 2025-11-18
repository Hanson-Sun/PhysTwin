import open3d as o3d
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import sys
import scipy.linalg
import json
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dust3r'))
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def get_scaled_intrinsics(orig_intrinsic, orig_shape, out_shape):
    h_orig, w_orig = np.array(orig_shape).flatten()[:2]
    h_out, w_out = out_shape
    
    scale_x = w_out / w_orig
    scale_y = h_out / h_orig
    
    new_intrinsic = orig_intrinsic.copy()
    new_intrinsic[0, 0] *= scale_x
    new_intrinsic[1, 1] *= scale_y
    new_intrinsic[0, 2] *= scale_x
    new_intrinsic[1, 2] *= scale_y
    return new_intrinsic

def align_poses_similarity(pred_poses, gt_poses):
    """Compute similarity transform from predicted to GT using both positions and orientations"""
    # Extract camera centers
    P = pred_poses[:, :3, 3]
    Q = gt_poses[:, :3, 3]

    # Compute centroids
    muP, muQ = P.mean(axis=0), Q.mean(axis=0)
    P_c, Q_c = P - muP, Q - muQ

    # Compute scale from positions
    scale = np.sqrt(np.sum(Q_c**2) / np.sum(P_c**2))
    
    # Use camera orientations to compute rotation
    # Stack camera Z-axes (forward directions) and camera centers
    pred_dirs = pred_poses[:, :3, 2]  # Forward direction (Z-axis)
    gt_dirs = gt_poses[:, :3, 2]
    
    # Combine positions and orientations for better alignment
    P_combined = np.vstack([P_c, pred_dirs])
    Q_combined = np.vstack([Q_c, gt_dirs])
    
    # Compute rotation using SVD on combined data
    C = np.dot(Q_combined.T, P_combined)
    U, S, Vt = scipy.linalg.svd(C)
    R = np.dot(U, Vt)
    
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    # Compute translation
    t = muQ - scale * np.dot(R, muP)
    return R, t, scale

def project_points_to_camera(points_world, K, w2c, h, w):
    """Project 3D world points to camera and create depth map with better handling"""
    # Transform to camera space
    points_cam = (points_world @ w2c[:3, :3].T) + w2c[:3, 3]
    
    # Filter points behind camera
    valid_depth = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth]
    
    if len(points_cam) == 0:
        return np.zeros((h, w), dtype=np.float32)
    
    # Project to image plane
    points_2d_homog = points_cam @ K.T
    points_2d = points_2d_homog[:, :2] / points_2d_homog[:, 2:3]
    depths = points_cam[:, 2]
    
    # Filter points outside image bounds
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    
    x = x[valid]
    y = y[valid]
    depths = depths[valid]
    
    if len(x) == 0:
        return np.zeros((h, w), dtype=np.float32)
    
    # Convert to integer pixel coordinates
    x_int = np.round(x).astype(np.int32)
    y_int = np.round(y).astype(np.int32)
    
    # Clip to image bounds (safety)
    x_int = np.clip(x_int, 0, w - 1)
    y_int = np.clip(y_int, 0, h - 1)
    
    # Create depth map - use minimum depth for each pixel (closest point)
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    
    for xi, yi, di in zip(x_int, y_int, depths):
        depth_map[yi, xi] = min(depth_map[yi, xi], di)
    
    # Replace inf with 0
    depth_map[depth_map == np.inf] = 0
    
    return depth_map

def render_3d_scene(depths, intrinsics, extrinsics):
    pcds = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    for i in range(len(depths)):
        o3d_depth = o3d.geometry.Image(depths[i].astype(np.float32))
        K = intrinsics[i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depths[i].shape[1], depths[i].shape[0], 
            K[0,0], K[1,1], K[0,2], K[1,2]
        )
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth, intrinsic, 
            extrinsic=np.linalg.inv(extrinsics[i]), 
            depth_scale=1.0, depth_trunc=20.0
        )
        pcd.paint_uniform_color(colors[i % 3])
        pcds.append(pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    for pcd in pcds: 
        vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def visualize_depth(depth, vmin=None, vmax=None, percentiles=(2, 98)):
    """Return a BGR visualization of a single-channel depth map using OpenCV colormaps.
    - depth: 2D numpy array (float), depth in meters (zeros or NaN considered invalid)
    - vmin/vmax: optional manual min/max in same units as depth. If None, computed from data
      using the provided percentile range to reduce outlier impact.
    - percentiles: tuple for lower/upper percentiles when vmin/vmax are None.

    Returns a uint8 BGR image (H,W,3).
    """
    if depth is None:
        raise ValueError('depth must be a numpy array')
    d = np.array(depth, copy=False)
    if d.ndim != 2:
        raise ValueError('depth must be 2D')

    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        h, w = d.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    vals = d[valid]
    if vmin is None or vmax is None:
        lo, hi = np.percentile(vals, [percentiles[0], percentiles[1]])
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
    # avoid degenerate range
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Normalize to 0-255
    norm = (d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)

    # Choose colormap
    cmap_attr = 'COLORMAP_TURBO' if hasattr(cv2, 'COLORMAP_TURBO') else 'COLORMAP_MAGMA'
    try:
        cmap = getattr(cv2, cmap_attr)
    except Exception:
        cmap = cv2.COLORMAP_JET

    colored = cv2.applyColorMap(gray, cmap)

    # Make invalid pixels black
    if not np.all(valid):
        mask = (~valid).astype(np.uint8)
        colored[mask == 1] = (0, 0, 0)

    return colored

def show_depths_blocking(depths, names=None):
    """Show multiple depth visualizations in OpenCV windows and block until closed by the user.

    depths: list of 2D depth arrays
    names: optional list of window titles
    """
    if names is None:
        names = [f'Depth cam {i}' for i in range(len(depths))]

    imgs = []
    for d in depths:
        try:
            imgs.append(visualize_depth(d))
        except Exception:
            # fallback to a black image
            h, w = d.shape if (hasattr(d, 'shape') and len(d.shape) == 2) else (480, 640)
            imgs.append(np.zeros((h, w, 3), dtype=np.uint8))

    for name, img in zip(names, imgs):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)

    # Block until all windows closed or ESC pressed
    try:
        while True:
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                break
            # Check if any window is still visible
            alive = False
            try:
                for name in names:
                    prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
                    if prop > 0:
                        alive = True
                        break
            except Exception:
                # If getWindowProperty not available, fallback to single waitKey
                pass
            if not alive:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for name in names:
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='Metric Point Map Generator')
    parser.add_argument('--case_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.case_dir, 'infer_depth')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration
    with open(os.path.join(args.case_dir, 'calibrate.pkl'), 'rb') as f:
        gt_calib = pickle.load(f)
    
    with open(os.path.join(args.case_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Parse calibration
    if isinstance(gt_calib, list):
        calib_keys = list(range(len(gt_calib)))
        gt_poses_list = gt_calib
    else:
        calib_keys = sorted(gt_calib.keys())
        gt_poses_list = [gt_calib[k] for k in calib_keys]
    
    gt_intrinsics = [np.array(metadata['intrinsics'][i]) for i in range(len(calib_keys))]
    gt_poses_np = np.array(gt_poses_list)

    # Get frame lists
    cam_dirs = [os.path.join(args.case_dir, 'color', str(k)) for k in calib_keys]
    frame_files = [sorted(os.listdir(d)) for d in cam_dirs]
    common_frames = sorted(list(set(frame_files[0]) & set(frame_files[1]) & set(frame_files[2])))

    # Load model
    model = AsymmetricCroCo3DStereo.from_pretrained(
        'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
    ).to(args.device)

    for frame_idx, frame_name in enumerate(tqdm(common_frames, desc="Processing")):
        img_paths = [os.path.join(cam_dirs[i], frame_name) for i in range(len(calib_keys))]
        images = load_images(img_paths, size=512)

        # Run DUSt3R
        output = inference(
            make_pairs(images, scene_graph='complete', symmetrize=True), 
            model, args.device
        )
        scene = global_aligner(
            output, device=args.device, 
            mode=GlobalAlignerMode.PointCloudOptimizer
        )
        scene.compute_global_alignment(init="mst", niter=300)

        # Get predicted poses and point clouds
        pred_poses = scene.get_im_poses().detach().cpu().numpy()
        pts3d = scene.get_pts3d()

        if args.visualize and frame_idx == 0:
            scene.show()
        
        # Step 1: Build global point cloud in DUSt3R's predicted world space
        all_points_pred_world = []
        for i in range(len(calib_keys)):
            pts_cam = pts3d[i].detach().cpu().numpy()
            h, w = pts_cam.shape[:2]
            pts_flat = pts_cam.reshape(-1, 3)
            
            # Transform from camera to predicted world
            pred_c2w = pred_poses[i]
            pts_world = (pts_flat @ pred_c2w[:3, :3].T) + pred_c2w[:3, 3]
            all_points_pred_world.append(pts_world)
        
        all_points_pred_world = np.vstack(all_points_pred_world)
        
        # Step 2: Align predicted world to GT world using Procrustes
        R_align, t_align, scale = align_poses_similarity(pred_poses, gt_poses_np)
        all_points_gt_world = scale * (all_points_pred_world @ R_align.T) + t_align
        
        # Step 3: Project aligned points onto each GT camera
        current_frame_depths = []
        scaled_intrinsics = []
        
        for i, k in enumerate(calib_keys):
            h, w = pts3d[i].shape[:2]
            
            # Get GT camera parameters
            gt_w2c = np.linalg.inv(gt_poses_list[i])
            orig_k = gt_intrinsics[i]
            scaled_k = get_scaled_intrinsics(orig_k, images[i]['true_shape'], (h, w))
            
            # Project global point cloud onto this camera
            depth_map = project_points_to_camera(all_points_gt_world, scaled_k, gt_w2c, h, w)
            
            # Save
            save_path = os.path.join(output_dir, str(k))
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, f"{os.path.splitext(frame_name)[0]}.npy"), depth_map)
            
            current_frame_depths.append(depth_map)
            scaled_intrinsics.append(scaled_k)

        if args.visualize:
            try:
                names = [f'cam_{k}' for k in calib_keys]
                show_depths_blocking(current_frame_depths, names=names)
            except Exception as e:
                print(f'Could not show depth visualizations: {e}')

        if args.visualize and frame_idx == 0:
            render_3d_scene(current_frame_depths, scaled_intrinsics, gt_poses_list)

if __name__ == "__main__":
    main()