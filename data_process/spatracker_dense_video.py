"""
Per-camera 3D point tracking using SpaTrackerV2.
Tracks objects in 3D space by processing each camera independently.
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
import time
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from dense_track_utils import (
    exist_dir, read_mask, resize_mask, load_video_frames, 
    load_depth_frames, load_camera_params, load_separated_masks,
    resize_video_and_depth
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--case_name", type=str, required=True)

# Only parse if this is the main module
_args = None
if __name__ == "__main__":
    _args = parser.parse_args()
    base_path = _args.base_path
    case_name = _args.case_name
else:
    # Will be set by dense_track.py
    base_path = None
    case_name = None
num_cam = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {device}")

def load_spatracker_model(window_size=None, overlap=None):
    """Load SpaTrackerV2 model with proper temporal windowing support.
    
    TEMPORAL CONSISTENCY: Cache propagation across windows requires overlap > 0.
    
    VRAM CONSTRAINTS FOR 12GB GPU:
    - window_size=80+: May cause OOM during transformer cross-attention
    - window_size=60: Official eval size, should fit for most resolutions
    - window_size=40: Conservative size, guaranteed to fit
    - window_size=20: Extreme memory savings, many windows but still has cache
    
    Args:
        window_size: Frames per window (default 500 for offline model).
                    For 12GB GPU: use 60, 40, or 20 depending on resolution.
                    Memory scales ~quadratically with window_size.
        overlap: Overlap between windows (frames). Use 3-6 for small windows.
                **CRITICAL: Keep overlap > 0 to enable cache propagation!**
                Typical: 5% of window_size (e.g., 3 for window_size=60).
    
    Returns:
        Loaded SpaTrackerV2 model with cache-based temporal consistency enabled
    """
    logger.info("Loading SpaTrackerV2 model...")
    
    # Add SpaTrackerV2 root to sys.path so internal imports work
    spatracker_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SpaTrackerV2")
    
    if spatracker_root not in sys.path:
        sys.path.insert(0, spatracker_root)
    
    # Import Predictor
    from models.SpaTrackV2.models.predictor import Predictor
    
    # Load the offline model
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.eval()
    model.to(device)
    
    # Override window size and overlap for memory efficiency if specified
    if window_size is not None:
        logger.info(f"  Overriding window_size: {model.S_wind} → {window_size}")
        model.S_wind = window_size
    
    if overlap is not None:
        logger.info(f"  Overriding overlap: {model.overlap} → {overlap}")
        model.overlap = overlap
    
    logger.info(f"✓ SpaTrackerV2 model loaded (window_size={model.S_wind}, overlap={model.overlap})")
    
    overlap_ratio = (model.overlap / model.S_wind) * 100 if model.S_wind > 0 else 0
    logger.info(f"  Overlap ratio: {overlap_ratio:.1f}%")
    
    if model.overlap == 0:
        logger.warning(f"  ⚠ WARNING: overlap=0 disables temporal feature propagation!")
        logger.warning(f"             Cache cannot flow across windows. Use overlap > 0.")
    
    return model


def generate_query_points_2d_from_mask(mask, num_points=50):
    """
    Generate 2D query points from mask.
    
    Args:
        mask: [H, W] binary mask
        num_points: target number of points
    
    Returns:
        query_points_2d: [N, 3] format [frame_idx=0, x, y]
    """
    y, x = np.where(mask)
    
    if len(y) == 0:
        logger.warning("Mask is empty! Returning empty points.")
        return np.zeros((0, 3), dtype=np.float32)
    
    # Downsample to target number if needed
    if len(y) > num_points:
        indices = np.random.choice(len(y), num_points, replace=False)
        x, y = x[indices], y[indices]
    
    # Format as [frame_idx=0, x_pixel, y_pixel]
    points_2d = np.stack([np.zeros(len(x)), x.astype(np.float32), y.astype(np.float32)], axis=1)
    
    logger.info(f"  Generated {len(points_2d)} query points from mask")
    return points_2d


def run_spatracker_inference_single_camera(model, video, depth, intrs, extrs, object_mask, controller_mask):
    """
    Run SpaTrackerV2 inference for a single camera using native 3D queries.
    
    Args:
        model: SpaTrackerV2 Predictor model
        video: [T, H, W, 3] video frames in range [0, 255]
        depth: [T, H, W] depth maps
        intrs: [3, 3] intrinsic matrix (single camera)
        extrs: [4, 4] extrinsic matrix (camera-to-world)
        object_mask: [H, W] binary mask for object
        controller_mask: [H, W] binary mask for controller
    
    Returns:
        object_tracks: [T, N_obj, 3] 3D coordinates in world space
        object_vis: [T, N_obj] visibility
        controller_tracks: [T, N_ctrl, 3] 3D coordinates in world space
        controller_vis: [T, N_ctrl] visibility
    """
    logger.info("  Running SpaTrackerV2 inference (3D native tracking)...")
    logger.info(f"    Input shapes: video={video.shape}, depth={depth.shape}")
    
    # Ensure video and depth have the same number of frames
    T_video = video.shape[0]
    T_depth = depth.shape[0]
    
    if T_video != T_depth:
        logger.warning(f"    Frame count mismatch: video={T_video}, depth={T_depth}")
        min_frames = min(T_video, T_depth)
        video = video[:min_frames]
        depth = depth[:min_frames]
        logger.info(f"    Synced to {min_frames} frames")
    else:
        logger.info(f"    Frame counts match: {T_video} frames")
    
    T, H, W, C = video.shape
    
    # Convert video from [T, H, W, 3] to [T, 3, H, W] format expected by model
    # Follow official pattern: torch.from_numpy() then permute then .float()
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # [T, 3, H, W] as float32
    # Pass depth as numpy array (official code uses numpy arrays)
    depth_array = depth.astype(np.float32)  # Keep as numpy [T, H, W]
    
    # Prepare camera parameters as numpy arrays (official code uses numpy)
    intrs_array = np.tile(intrs[None], (T, 1, 1)).astype(np.float32)  # [T, 3, 3]
    extrs_array = np.tile(extrs[None], (T, 1, 1)).astype(np.float32)  # [T, 4, 4]
    
    # Generate 2D query points from masks (model lifts to 3D internally using depth)
    object_queries_2d = generate_query_points_2d_from_mask(object_mask, num_points=500)
    controller_queries_2d = generate_query_points_2d_from_mask(controller_mask, num_points=100)
    
    object_queries_array = object_queries_2d
    controller_queries_array = controller_queries_2d
    
    with torch.no_grad():
        # Run inference for object points with 2D queries
        if object_queries_array is not None and len(object_queries_array) > 0:
            logger.info("    Tracking object points...")
            logger.info(f"    Model config: window_size={model.S_wind}, overlap={model.overlap}")
            logger.info(f"    Video length: {T} frames")
            start_time = time.time()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Use bfloat16 mixed precision for memory efficiency (matches official code)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                (c2w_traj, intrs_out, point_map, conf_depth,
                 track3d_pred_obj, track2d_pred_obj, vis_pred_obj, conf_pred_obj, 
                 _) = model.forward(
                    video_tensor, depth=depth_array,
                    intrs=intrs_array, extrs=extrs_array,
                    queries=object_queries_array,
                    fps=1, full_point=False, iters_track=4,
                    query_no_BA=True, fixed_cam=False, stage=1, 
                    support_frame=T-1, replace_ratio=0.2
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            logger.info(f"    ✓ Object tracking done ({elapsed:.1f}s)")
            
            # Transform 3D tracks to world space using GROUND TRUTH extrinsics
            # (NOT model-estimated c2w_traj, which won't align across cameras)
            device = track3d_pred_obj.device
            # Tile extrinsics to match time dimension [4, 4] -> [T, 4, 4]
            extrs_c2w = torch.from_numpy(np.tile(extrs[None], (T, 1, 1))).to(device).float()
            object_tracks = (torch.einsum("tij,tnj->tni", extrs_c2w[:,:3,:3], 
                                          track3d_pred_obj[:,:,:3]) + 
                            extrs_c2w[:,:3,3][:,None,:]).cpu().numpy()  # [T, N, 3]
            # Squeeze visibility to ensure 2D [T, N] shape
            object_vis = vis_pred_obj.cpu().numpy()  # [T, N] or [T, N, 1]
            if object_vis.ndim == 3:
                object_vis = np.squeeze(object_vis, axis=-1)
            logger.info(f"    Object: tracks shape {object_tracks.shape}, range [{object_tracks.min():.3f}, {object_tracks.max():.3f}]")
        else:
            logger.info("    Skipping object tracking (empty mask)")
            object_tracks = np.zeros((T, 0, 3), dtype=np.float32)
            object_vis = np.zeros((T, 0), dtype=np.float32)
        
        # Run inference for controller points with 2D queries
        if controller_queries_array is not None and len(controller_queries_array) > 0:
            logger.info("    Tracking controller points...")
            start_time = time.time()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Use bfloat16 mixed precision for memory efficiency (matches official code)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                (c2w_traj, _, _, _,
                 track3d_pred_ctrl, track2d_pred_ctrl, vis_pred_ctrl, _, 
                 _) = model.forward(
                    video_tensor, depth=depth_array,
                    intrs=intrs_array, extrs=extrs_array,
                    queries=controller_queries_array,
                    fps=1, full_point=False, iters_track=4,
                    query_no_BA=True, fixed_cam=False, stage=1,
                    support_frame=T-1, replace_ratio=0.2
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            logger.info(f"    ✓ Controller tracking done ({elapsed:.1f}s)")
            
            # Transform 3D tracks to world space using GROUND TRUTH extrinsics
            # (NOT model-estimated c2w_traj, which won't align across cameras)
            device = track3d_pred_ctrl.device
            # Tile extrinsics to match time dimension [4, 4] -> [T, 4, 4]
            extrs_c2w = torch.from_numpy(np.tile(extrs[None], (T, 1, 1))).to(device).float()
            controller_tracks = (torch.einsum("tij,tnj->tni", extrs_c2w[:,:3,:3],
                                             track3d_pred_ctrl[:,:,:3]) +
                               extrs_c2w[:,:3,3][:,None,:]).cpu().numpy()  # [T, N, 3]
            # Squeeze visibility to ensure 2D [T, N] shape
            controller_vis = vis_pred_ctrl.cpu().numpy()  # [T, N] or [T, N, 1]
            if controller_vis.ndim == 3:
                controller_vis = np.squeeze(controller_vis, axis=-1)
        else:
            logger.info("    Skipping controller tracking (empty mask)")
            controller_tracks = np.zeros((T, 0, 3), dtype=np.float32)
            controller_vis = np.zeros((T, 0), dtype=np.float32)
    
    logger.info(f"✓ Object: {object_tracks.shape[1]} points, Controller: {controller_tracks.shape[1]} points")
    return object_tracks, object_vis, controller_tracks, controller_vis


def save_spatracker_output(base_path, case_name, cam_id, object_tracks, object_vis, 
                          controller_tracks, controller_vis):
    """
    Save SpaTrackerV2 outputs per camera for compatibility with rest of pipeline.
    """
    output_dir = f"{base_path}/{case_name}/spatracker"
    exist_dir(output_dir)
    
    # Save separate camera output
    cam_dir = f"{output_dir}/camera_{cam_id}"
    exist_dir(cam_dir)
    
    object_path = f"{cam_dir}/object_tracks.npz"
    np.savez(
        object_path,
        tracks=object_tracks,
        visibility=object_vis,
    )
    
    controller_path = f"{cam_dir}/controller_tracks.npz"
    np.savez(
        controller_path,
        tracks=controller_tracks,
        visibility=controller_vis,
    )
    
    # Also save combined for backward compatibility
    combined_tracks = np.concatenate([object_tracks, controller_tracks], axis=1)
    combined_vis = np.concatenate([object_vis, controller_vis], axis=1)
    
    combined_path = f"{cam_dir}/combined_tracks.npz"
    np.savez(
        combined_path,
        tracks=combined_tracks,
        visibility=combined_vis,
    )
    
    logger.info(f"✓ Camera {cam_id}: {object_path}")


def main():
    logger.info(f"\n{'='*60}")
    logger.info(f"SpaTrackerV2 Dense Video Tracking: {case_name}")
    logger.info(f"{'='*60}\n")
    logger.info("Step 1: Loading RGB videos...")
    rgbs = []
    video_height, video_width = None, None
    
    for cam_id in range(num_cam):
        video_path = f"{base_path}/{case_name}/color/{cam_id}.mp4"
        frames = load_video_frames(video_path)
        rgbs.append(frames)
        
        if video_height is None:
            video_height, video_width = frames.shape[1:3]
    
    # Load depth maps in native resolution
    logger.info("Step 2: Loading depth maps...")
    depths = []
    native_depth_shapes = []
    
    for cam_id in range(num_cam):
        depth_dir = f"{base_path}/{case_name}/depth/{cam_id}"
        # Load without resizing to preserve native resolution
        depth_frames = load_depth_frames(depth_dir, target_shape=None)
        
        if depth_frames is None:
            logger.error(f"✗ Failed to load depth for camera {cam_id}")
            sys.exit(1)
        
        depths.append(depth_frames)
        native_depth_shapes.append(depth_frames.shape[1:])
    
    # Load camera parameters
    logger.info("Step 3: Loading camera parameters...")
    intrs, extrs = load_camera_params(base_path, case_name)
    
    # Auto-resize RGB videos to match depth resolution
    logger.info("Step 3b: Resizing RGB videos to match depth resolution...")
    rgbs_resized = []
    depths_resized = []
    intrs_resized = []
    
    for cam_id in range(num_cam):
        depth_h, depth_w = native_depth_shapes[cam_id]
        video_h, video_w = rgbs[cam_id].shape[1:3]
        
        logger.info(f"  Camera {cam_id}: video frames={rgbs[cam_id].shape[0]}, depth frames={depths[cam_id].shape[0]}")
        
        if (depth_h, depth_w) != (video_h, video_w):
            logger.info(f"    Spatial resize: {video_w}x{video_h} → {depth_w}x{depth_h}")
            video_r, depth_r, intr_r = resize_video_and_depth(rgbs[cam_id], depths[cam_id], depth_h, intrs[cam_id])
            rgbs_resized.append(video_r)
            depths_resized.append(depth_r)
            intrs_resized.append(intr_r)
            logger.info(f"    After resize: video frames={video_r.shape[0]}, depth frames={depth_r.shape[0]}")
        else:
            logger.info(f"    Already {depth_w}x{depth_h} (no resize needed)")
            rgbs_resized.append(rgbs[cam_id])
            depths_resized.append(depths[cam_id])
            intrs_resized.append(intrs[cam_id])
    
    rgbs = rgbs_resized
    depths = depths_resized
    intrs = np.stack(intrs_resized)
    video_height, video_width = native_depth_shapes[0]
    
    # Load masks
    logger.info("Step 4: Loading masks...")
    object_masks, controller_masks, mask_info = load_separated_masks(
        base_path, case_name, num_cam, video_height, video_width
    )
    
    # Load model
    logger.info("Step 5: Loading SpaTrackerV2 model...")
    model = load_spatracker_model(window_size=35, overlap=2)
    
    # Process each camera
    logger.info("Step 6: Running inference per camera...")
    for cam_id in tqdm(range(num_cam), desc="Processing cameras"):
        logger.info(f"\nProcessing camera {cam_id}...")
        
        video = rgbs[cam_id].astype(np.uint8)  # [T, H, W, 3]
        depth = depths[cam_id]  # [T, H, W]
        
        # Use camera-specific intrinsics and extrinsics
        intr = intrs[cam_id]  # [3, 3]
        extr = extrs[cam_id]  # [4, 4]
        
        object_tracks, object_vis, controller_tracks, controller_vis = \
            run_spatracker_inference_single_camera(
                model, video, depth, intr, extr,
                object_masks[cam_id], controller_masks[cam_id]
            )
        
        save_spatracker_output(
            base_path, case_name, cam_id,
            object_tracks, object_vis,
            controller_tracks, controller_vis
        )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ SpaTrackerV2 tracking complete for {case_name}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
