#!/usr/bin/env python3
import argparse
import json
import pickle
import os
from itertools import islice
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Integrating with your existing framework
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from qqtt.utils import logger, cfg

# ----------------------------
# IO and Config Bridge
# ----------------------------

def load_trajectory(path: Path, key: str, fallback_keys=()):
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        raw = data.get(key)
        if raw is None:
            for fallback in fallback_keys:
                if fallback in data:
                    raw = data[fallback]
                    break
        if raw is None:
            raise KeyError(f"No key '{key}' found in {path}. Available: {list(data.keys())}")
    else:
        raw = data

    return np.asarray(raw, dtype=np.float32)

def setup_global_cfg(args):
    """
    Populates the singleton cfg object so that it matches 
    the internal state expected by qqtt modules.
    """
    # 1. Load Camera Calibration
    with open(args.calibrate_pkl, "rb") as f:
        c2ws = np.asarray(pickle.load(f), dtype=np.float64)

    # Apply the coordinate flip if required by the config logic
    # This aligns the NeRF/OpenCV camera with Open3D's coordinate system
    flip_mat = np.eye(4)
    flip_mat[1, 1] = -1
    flip_mat[2, 2] = -1
    # if args.reverse_z:
    #     c2ws = c2ws @ flip_mat

    # Store in cfg
    cfg.c2ws = c2ws
    cfg.w2cs = np.linalg.inv(c2ws)

    # 2. Load Metadata
    with open(args.metadata_json, "r") as f:
        meta = json.load(f)
    
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]
    
    # 3. Set Overlay Path
    if args.overlay_dir:
        cfg.overlay_path = args.overlay_dir
    else:
        cfg.overlay_path = f"{args.base_path}/{args.case_name}/color"

    logger.info(f"Global CFG initialized for case: {args.case_name}")

# ----------------------------
# Geometry and Rendering
# ----------------------------

def build_camera_params(width, height, K, ext):
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.set_intrinsics(width, height, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = cam
    params.extrinsic = np.asarray(ext, dtype=np.float64)
    return params

def composite_overlay(bg, render):
    # Mask where the render is NOT the black background
    mask = np.any(render > 0, axis=2)
    out = bg.copy()
    # Resize bg if it doesn't match render (safety check)
    if bg.shape[:2] != render.shape[:2]:
        import cv2
        bg = cv2.resize(bg, (render.shape[1], render.shape[0]))
    
    out[mask] = render[mask]
    return out

# ----------------------------
# Main Execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--object_pkl", required=True)
    parser.add_argument("--controller_source_pkl", required=True)
    parser.add_argument("--object_key", default="object_points")
    parser.add_argument("--controller_key", default="controller_points")
    parser.add_argument("--calibrate_pkl", required=True)
    parser.add_argument("--metadata_json", required=True)
    parser.add_argument("--overlay_dir", default=None)
    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--controller_size", type=float, default=0.012)
    parser.add_argument("--output", default="inference_render.mp4")
    parser.add_argument("--reverse_z", action="store_true", help="Flip the Z-axis of the object and controller points")
    args = parser.parse_args()

    # 1. Initialize the global config singleton
    setup_global_cfg(args)

    # 2. Load trajectories
    object_traj = load_trajectory(Path(args.object_pkl), args.object_key)
    controller_traj = load_trajectory(Path(args.controller_source_pkl), args.controller_key)

    if args.reverse_z:
        logger.info("Reversing Z-axis for point trajectories.")
        object_traj[..., 2] *= -1
        # controller_traj[..., 2] *= -1
    
    width, height = cfg.WH
    num_frames = min(len(object_traj), len(controller_traj))

    # 3. Setup Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = args.point_size

    # Prepare Geometries
    obj_pcd = o3d.geometry.PointCloud()
    # Color by height (Y-axis)
    y_vals = object_traj[0, :, 1]
    y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8)
    obj_colors = plt.cm.turbo(y_norm)[:, :3]

    # Initialize controller spheres
    ctrl_count = controller_traj.shape[1]
    ctrl_meshes = []
    for _ in range(ctrl_count):
        m = o3d.geometry.TriangleMesh.create_sphere(radius=args.controller_size)
        m.paint_uniform_color([1.0, 0.1, 0.1]) # Reddish
        ctrl_meshes.append(m)

    # Setup Video Writer
    writer = imageio.get_writer(args.output, fps=args.fps)
    cam_params = build_camera_params(width, height, cfg.intrinsics[args.view_index], cfg.w2cs[args.view_index])

    logger.info(f"Starting render for {num_frames} frames...")

    try:
        for i in range(num_frames):
            # Update Object PCD
            obj_pcd.points = o3d.utility.Vector3dVector(object_traj[i])
            obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)

            if i == 0:
                vis.add_geometry(obj_pcd)
                for m in ctrl_meshes:
                    vis.add_geometry(m)
            else:
                vis.update_geometry(obj_pcd)
                for j in range(ctrl_count):
                    # Move spheres to absolute world positions
                    ctrl_meshes[j].translate(controller_traj[i, j], relative=False)
                    vis.update_geometry(ctrl_meshes[j])

            # Force camera viewpoint
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            
            vis.poll_events()
            vis.update_renderer()

            # Capture & Composite
            render_rgb = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
            
            bg_path = Path(cfg.overlay_path) / str(args.view_index) / f"{i}.png"
            if bg_path.exists():
                bg_img = imageio.imread(bg_path)
                final_frame = composite_overlay(bg_img, render_rgb)
            else:
                final_frame = render_rgb

            writer.append_data(final_frame)
            
            if i % 50 == 0:
                logger.info(f"Rendered frame {i}/{num_frames}")

    finally:
        writer.close()
        vis.destroy_window()
        logger.info(f"Rendering complete: {args.output}")

if __name__ == "__main__":
    main()