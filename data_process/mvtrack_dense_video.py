"""
Multi-view 3D point tracking using MVTracker.
Replaces monocular CoTracker with multi-view MVTracker for consistent cross-view tracking.
"""

import torch
import imageio.v3 as iio
import glob
import cv2
import numpy as np
import os
import json
import pickle
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import rerun as rr

from dense_track_utils import (
    exist_dir, read_mask, resize_mask, load_video_frames, 
    load_depth_frames, load_camera_params, load_separated_masks
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
num_cam = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {device}")


def load_mvtracker_model():
    """Load the MVTracker model from torch hub."""
    logger.info("Loading MVTracker model...")
    # Ensure mvtracker directory is in sys.path for torch.hub.load()
    mvtracker_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mvtracker")
    if mvtracker_dir not in sys.path:
        sys.path.insert(0, mvtracker_dir)
    
    mvtracker = torch.hub.load(
        "ethz-vlg/mvtracker",
        "mvtracker",
        pretrained=True,
        device=device
    )
    mvtracker.eval()
    logger.info("✓ MVTracker model loaded successfully")
    return mvtracker





def run_mvtracker_inference(mvtracker, rgbs, depths, intrs, extrs, query_points_3d):
    """
    Run MVTracker inference with sliding window processing.
    
    Args:
        rgbs: [V, T, H, W, 3] in range [0, 255]
        depths: [V, T, H, W]
        intrs: [V, 3, 3]
        extrs: [V, 4, 4] camera-to-world matrices
        query_points_3d: [N, 3] 3D points in camera 0 space (MVTracker handles multi-camera fusion)
    
    Returns:
        pred_tracks: Predicted tracks (T, N, 3) in camera 0 coordinate space
        pred_vis: Visibility mask (T, N)
    """
    logger.info("Running MVTracker inference...")
    
    V, T, H, W, C = rgbs.shape
    N = query_points_3d.shape[0]
    
    # Process full sequence
    
    # Reshape for MVTracker: [V, T, H, W, 3] -> [1, V, T, 3, H, W]
    rgbs_mvtrack = rgbs.transpose(0, 1, 4, 2, 3)[None]  # [1, V, T, 3, H, W]
    
    # Reshape depths: [B, V, T, 1, H, W]
    depths_mvtrack = depths[None, :, :, None, :, :]  # [1, V, T, 1, H, W]
    
    # Expand intrinsics to per-frame: [V, 3, 3] -> [B, V, T, 3, 3]
    intrs_mvtrack = np.tile(intrs[np.newaxis, :, np.newaxis, :, :], (1, 1, T, 1, 1))
    
    # Convert extrinsics: c2w -> w2c
    extrs_w2c = []
    for c2w in extrs:
        w2c = np.linalg.inv(c2w)
        extrs_w2c.append(w2c)
    extrs_w2c = np.array(extrs_w2c, dtype=np.float32)
    
    extrs_3x4 = extrs_w2c[:, :3, :]
    extrs_mvtrack = np.tile(extrs_3x4[np.newaxis, :, np.newaxis, :, :], (1, 1, T, 1, 1))
    
    # Create query points in MVTracker format: [B, N, 4] where first dim is timestamp
    query_points_mvtrack = np.zeros((1, N, 4), dtype=np.float32)
    query_points_mvtrack[0, :, 0] = 0  # All query points start at frame 0
    query_points_mvtrack[0, :, 1:] = query_points_3d  # [x, y, z]
    query_points_mvtrack = torch.from_numpy(query_points_mvtrack).float().to(device)
    
    with torch.no_grad():
        results = mvtracker(
            rgbs=torch.from_numpy(rgbs_mvtrack).float().to(device) / 255.0,
            depths=torch.from_numpy(depths_mvtrack).float().to(device),
            intrs=torch.from_numpy(intrs_mvtrack).float().to(device),
            extrs=torch.from_numpy(extrs_mvtrack).float().to(device),
            query_points_3d=query_points_mvtrack,
        )
        
        pred_tracks = results["traj_e"][0].cpu().numpy()     # [T, N, 3]
        pred_vis = results["vis_e"][0].cpu().numpy()         # [T, N]
    
    logger.info(f"✓ MVTracker inference complete: tracks shape {pred_tracks.shape}")
    return pred_tracks, pred_vis


def save_mvtracker_output(base_path, case_name, object_tracks, object_vis, controller_tracks, controller_vis, num_cam=3):
    """
    Save MVTracker outputs (object and controller) separately for compatibility with data_process_track.py.
    Saves as per-camera NPZ files.
    
    Args:
        object_tracks: Object coordinates (T, N_obj, 3) in 3D world space
        object_vis: Object visibility (T, N_obj)
        controller_tracks: Controller coordinates (T, N_ctrl, 3) in 3D world space
        controller_vis: Controller visibility (T, N_ctrl)
        num_cam: Number of cameras (for compatibility format)
    """
    output_dir = f"{base_path}/{case_name}/mvtrack"
    exist_dir(output_dir)
    
    # Save object and controller tracks in separate files with metadata
    object_path = f"{output_dir}/object_tracks.npz"
    np.savez(
        object_path,
        tracks=object_tracks,  # [T, N_obj, 3]
        visibility=object_vis,  # [T, N_obj]
    )
    logger.info(f"✓ Saved object tracks ({object_tracks.shape[1]} points): {object_path}")
    
    controller_path = f"{output_dir}/controller_tracks.npz"
    np.savez(
        controller_path,
        tracks=controller_tracks,  # [T, N_ctrl, 3]
        visibility=controller_vis,  # [T, N_ctrl]
    )
    logger.info(f"✓ Saved controller tracks ({controller_tracks.shape[1]} points): {controller_path}")
    
    # Also save combined for backward compatibility (each camera gets copies)
    # Concatenate object and controller (object first, then controller)
    combined_tracks = np.concatenate([object_tracks, controller_tracks], axis=1)
    combined_vis = np.concatenate([object_vis, controller_vis], axis=1)
    
    for cam_id in range(num_cam):
        cam_path = f"{output_dir}/{cam_id}.npz"
        np.savez(
            cam_path,
            tracks=combined_tracks,  # [T, N_obj + N_ctrl, 3]
            visibility=combined_vis,  # [T, N_obj + N_ctrl]
        )
    
    logger.info(f"✓ Saved combined tracks to {output_dir}/{{0,1,2}}.npz")





def main():
    logger.info(f"\n{'='*60}")
    logger.info(f"MVTracker Dense Video Tracking: {case_name}")
    logger.info(f"{'='*60}\n")
    
    # ============================================================================
    # 1. Load RGB videos from all cameras
    # ============================================================================
    logger.info("Step 1: Loading RGB videos...")
    rgbs = []
    video_height, video_width = None, None
    
    for cam_id in range(num_cam):
        video_path = f"{base_path}/{case_name}/color/{cam_id}.mp4"
        frames = load_video_frames(video_path)
        rgbs.append(frames)
        
        if video_height is None:
            video_height, video_width = frames.shape[1:3]
    
    rgbs = np.stack(rgbs)  # [V, T, H, W, 3]
    logger.info(f"✓ RGB videos shape: {rgbs.shape}")
    
    # Sync frame counts across cameras
    min_frames = min(rgb.shape[0] for rgb in [rgbs[i] for i in range(num_cam)])
    rgbs = rgbs[:, :min_frames]
    
    # ============================================================================
    # 2. Load depth maps from all cameras
    # ============================================================================
    logger.info("Step 2: Loading depth maps...")
    depths = []
    all_depths_loaded = True
    
    for cam_id in range(num_cam):
        depth_dir = f"{base_path}/{case_name}/depth/{cam_id}"
        # Try to load depth at original RGB resolution
        depth_frames = load_depth_frames(depth_dir, target_shape=(video_width, video_height))
        
        if depth_frames is None:
            logger.error(f"✗ Failed to load depth for camera {cam_id}")
            all_depths_loaded = False
            break
        
        depths.append(depth_frames)
    
    # If depth maps missing, fail with helpful error
    if not all_depths_loaded or len(depths) != num_cam:
        logger.error(f"\n{'='*60}")
        logger.error("✗ DEPTH MAP ERROR")
        logger.error(f"{'='*60}")
        logger.error(f"Could not load all depth maps for case: {case_name}")
        logger.error(f"Expected: {num_cam} cameras with depth maps at {base_path}/{case_name}/depth/{{0,1,2}}/")
        logger.error(f"\nLikely causes:")
        logger.error(f"  - Depth maps not yet generated (run depth inference first)")
        logger.error(f"  - Depth maps at different location or format")
        logger.error(f"  - Calibration step not completed")
        logger.error(f"\nMVTracker requires depth maps. Please generate them before proceeding.")
        logger.error(f"{'='*60}\n")
        sys.exit(1)
    
    depths = np.stack(depths)  # [V, T, H, W]
    
    # Sync depth frame count with RGB
    min_frames = min(depths.shape[1], rgbs.shape[1])
    depths = depths[:, :min_frames]
    rgbs = rgbs[:, :min_frames]
    
    # ============================================================================
    # 3. Load camera parameters
    # ============================================================================
    logger.info("Step 3: Loading camera parameters...")
    intrs, extrs = load_camera_params(base_path, case_name)
    
    # ============================================================================
    # 4. Load and separate object/controller masks for query point generation
    # ============================================================================
    logger.info("Step 4: Loading masks...")
    object_masks, controller_masks, mask_info = load_separated_masks(
        base_path, case_name, num_cam, video_height, video_width
    )
    
    # ============================================================================
    # 5. Generate 3D query points for object and controller
    # ============================================================================
    logger.info("Step 5: Generating query points...")
    
    object_query_points_3d = generate_query_points_3d(depths[0, 0], intrs[0], object_masks[0])
    controller_query_points_3d = generate_query_points_3d(depths[0, 0], intrs[0], controller_masks[0])
    
    # ============================================================================
    # 5b. Query point coordinate system
    # ============================================================================
    # IMPORTANT: Do NOT transform query points to world space.
    # MVTracker internally handles the coordinate transformation using extrinsics.
    # Query points should be in camera 0 coordinate space.
    # (Removing the transform_points_cam_to_world call that was added earlier)
    # ============================================================================
    # 6. Load MVTracker model
    # ============================================================================
    logger.info("Step 6: Loading MVTracker model...")
    mvtracker = load_mvtracker_model()
    
    # ============================================================================
    # 7. Run MVTracker inference for object and controller separately
    # ============================================================================
    logger.info("Step 7: Running MVTracker inference...")
    
    object_tracks, object_vis = run_mvtracker_inference(
        mvtracker, rgbs, depths, intrs, extrs, object_query_points_3d
    )
    
    controller_tracks, controller_vis = run_mvtracker_inference(
        mvtracker, rgbs, depths, intrs, extrs, controller_query_points_3d
    )
    logger.info(f"✓ Tracked {object_tracks.shape[1]} object points and {controller_tracks.shape[1]} controller points\n")
    
    # ============================================================================
    # 7b. Transform tracks from camera 0 space to world space
    # ============================================================================
    logger.info("Step 7b: Transforming to world coordinate space...")
    
    # Use the same transformation as visualize_3d_scene.py
    c2w = extrs[0]  # [4, 4] camera-to-world for camera 0
    
    def transform_tracks_to_world(tracks, c2w):
        """Transform [T, N, 3] tracks from camera space to world space using c2w matrix."""
        T, N, _ = tracks.shape
        # Make homogeneous: [T*N, 4]
        tracks_homo = np.concatenate([tracks.reshape(T*N, 3), np.ones((T*N, 1))], axis=1)
        # Transform: [4, 4] @ [4, T*N] -> [4, T*N]
        tracks_world_homo = (c2w @ tracks_homo.T).T  # [T*N, 4]
        # Extract XYZ: [T*N, 3]
        tracks_world = tracks_world_homo[:, :3].reshape(T, N, 3)
        return tracks_world
    
    object_tracks_world = transform_tracks_to_world(object_tracks, c2w)
    controller_tracks_world = transform_tracks_to_world(controller_tracks, c2w)
    
    # ============================================================================
    # 8. Save outputs
    # ============================================================================
    logger.info("Step 8: Saving outputs...")
    save_mvtracker_output(
        base_path, case_name, 
        object_tracks_world, object_vis,
        controller_tracks_world, controller_vis,
        num_cam
    )
    
    # Save rerun visualization
    from mvtracker.utils.visualizer_rerun import log_tracks_to_rerun
    
    rr_filepath = f"{base_path}/{case_name}/mvtrack/mvtrack_viz.rrd"
    rr.init("mvtrack_dense_video", spawn=False)
    
    # Object tracks
    object_tracks_tensor = torch.from_numpy(object_tracks_world[np.newaxis]).float()
    object_vis_tensor = torch.from_numpy(object_vis[np.newaxis]).float()
    object_query_tensor = torch.from_numpy(np.concatenate([
        np.zeros((object_query_points_3d.shape[0], 1)), 
        object_query_points_3d
    ], axis=1)[np.newaxis]).float()
    
    log_tracks_to_rerun(
        dataset_name="mvtrack",
        datapoint_idx=case_name,
        predictor_name="mvtrack_object",
        gt_trajectories_3d_worldspace=None,
        gt_visibilities_any_view=None,
        query_points_3d=object_query_tensor,
        pred_trajectories=object_tracks_tensor,
        pred_visibilities=object_vis_tensor,
    )
    
    # Controller tracks
    controller_tracks_tensor = torch.from_numpy(controller_tracks_world[np.newaxis]).float()
    controller_vis_tensor = torch.from_numpy(controller_vis[np.newaxis]).float()
    controller_query_tensor = torch.from_numpy(np.concatenate([
        np.zeros((controller_query_points_3d.shape[0], 1)), 
        controller_query_points_3d
    ], axis=1)[np.newaxis]).float()
    
    log_tracks_to_rerun(
        dataset_name="mvtrack",
        datapoint_idx=case_name,
        predictor_name="mvtrack_controller",
        gt_trajectories_3d_worldspace=None,
        gt_visibilities_any_view=None,
        query_points_3d=controller_query_tensor,
        pred_trajectories=controller_tracks_tensor,
        pred_visibilities=controller_vis_tensor,
        method_id=1,
    )
    
    os.makedirs(os.path.dirname(rr_filepath), exist_ok=True)
    rr.save(rr_filepath)
    logger.info(f"✓ Saved rerun visualization to {rr_filepath}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ MVTracker tracking complete for {case_name}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
