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


def exist_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_mask(mask_path):
    """Load binary mask from image file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (mask > 0) if mask is not None else None


def resize_mask(mask, target_shape):
    """Resize mask to target (W, H) if needed."""
    if mask is None or (mask.shape[0], mask.shape[1]) == (target_shape[1], target_shape[0]):
        return mask
    return cv2.resize(mask.astype(np.uint8), target_shape, interpolation=cv2.INTER_NEAREST).astype(bool)


def load_video_frames(video_path):
    """Load RGB frames from video file."""
    try:
        frames = iio.imread(video_path, plugin="FFMPEG")
        return frames
    except Exception as e:
        logger.error(f"Failed to load video {video_path}: {e}")
        raise


def load_depth_frames(depth_dir, target_shape=None):
    """
    Load depth maps from directory of .npy files (sorted by frame number, not alphabetically).
    
    Args:
        depth_dir: Directory containing depth .npy files
        target_shape: (W, H) to resize depth maps to, if different from source
    
    Returns:
        depths: [T, H, W] depth maps, or None if not found
    """
    depth_files = glob.glob(f"{depth_dir}/*.npy")
    if not depth_files:
        logger.warning(f"No depth files found in {depth_dir}")
        return None
    
    # Sort by numeric filename, not alphabetically (0, 1, 2, ..., 10 not 0, 1, 10, 2, ...)
    depth_files = sorted(depth_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    depths = []
    for depth_file in depth_files:
        depth = np.load(depth_file).astype(np.float32)
        # Convert mm to meters if needed
        if np.nanmax(depth) > 20.0:
            depth = depth / 1000.0
        
        # Resize to target shape if needed
        # Note: target_shape is (W, H) for cv2.resize
        if target_shape is not None and depth.shape != (target_shape[1], target_shape[0]):
            depth = cv2.resize(depth, target_shape, interpolation=cv2.INTER_LINEAR)
        
        depths.append(depth)
    
    depths = np.stack(depths)  # [T, H, W]
    return depths


def load_camera_params(base_path, case_name):
    """
    Load camera intrinsics and extrinsics from calibration files.
    
    Returns:
        intrs: [V, 3, 3] camera intrinsic matrices
        extrs: [V, 4, 4] camera extrinsic matrices (c2w format)
    """
    case_path = f"{base_path}/{case_name}"
    
    # Load extrinsics (camera-to-world)
    try:
        with open(f"{case_path}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        extrs = np.array([np.array(c2w) for c2w in c2ws], dtype=np.float32)
        logger.info(f"✓ Loaded {extrs.shape[0]} camera-to-world matrices")
    except FileNotFoundError:
        logger.error(f"Missing {case_path}/calibrate.pkl")
        raise
    
    # Load intrinsics
    try:
        with open(f"{case_path}/metadata.json", "r") as f:
            metadata = json.load(f)
        intrs = np.array(metadata["intrinsics"], dtype=np.float32)
        logger.info(f"✓ Loaded {intrs.shape[0]} intrinsic matrices")
    except FileNotFoundError:
        logger.error(f"Missing {case_path}/metadata.json")
        raise
    except KeyError:
        logger.error(f"'intrinsics' key not found in metadata.json")
        raise
    
    return intrs, extrs


def generate_query_points_3d(depth, intrinsic, mask, num_points=1000):
    """
    Generate 3D query points from masked region using grid-based uniform sampling.
    
    Projects 2D masked pixels to 3D using depth and camera intrinsics.
    Uses grid-based sampling to ensure uniform distribution across the cloth surface.
    
    Args:
        depth: [H, W] single depth frame
        intrinsic: [3, 3] camera intrinsic matrix
        mask: [H, W] binary mask
        num_points: target number of points to sample
    
    Returns:
        query_points: [N, 3] 3D points in camera space [x, y, z]
    """
    # Ensure mask matches depth resolution
    if mask is not None and mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask > 0
    
    if mask is None:
        mask = depth > 0
    
    # Get pixel coordinates of masked region
    y, x = np.where(mask)
    
    if len(y) == 0:
        logger.warning("Mask is empty! Using full depth image.")
        y, x = np.where(depth > 0)
    
    # Unproject to camera space
    z = depth[y, x].astype(np.float32)
    
    # Remove invalid depths
    valid = z > 0
    y, x, z = y[valid], x[valid], z[valid]
    
    if len(z) == 0:
        logger.warning("No valid depth pixels found!")
        return np.array([[0, 0, 0.1]], dtype=np.float32)
    
    # Unproject using intrinsics
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy
    
    points_cam = np.stack([x_norm * z, y_norm * z, z], axis=1).astype(np.float32)  # [N, 3]
    
    # Uniform sampling using grid-based approach
    if len(points_cam) > num_points:
        # Grid-based downsampling: divide space into voxel grid and sample from each voxel
        # This ensures uniform distribution across the cloth
        grid_size = int(np.ceil(np.sqrt(len(points_cam) / num_points)))
        
        # Normalize to [0, 1] for grid binning
        min_coords = points_cam.min(axis=0)
        max_coords = points_cam.max(axis=0)
        scale = max_coords - min_coords
        scale[scale == 0] = 1.0  # avoid division by zero
        
        normalized = (points_cam - min_coords) / scale
        grid_indices = (normalized * grid_size).astype(int)
        
        # Sample one point per grid cell
        sampled_points = []
        for grid_idx in range(grid_size ** 3):
            mask_cell = np.all(grid_indices == np.unravel_index(grid_idx, (grid_size, grid_size, grid_size)), axis=1)
            if np.any(mask_cell):
                # Take centroid of points in this cell
                cell_points = points_cam[mask_cell]
                sampled_points.append(cell_points.mean(axis=0))
        
        if len(sampled_points) >= num_points:
            points_cam = np.array(sampled_points[:num_points])
        else:
            # Not enough cells, use farthest point sampling as fallback
            indices = np.random.choice(len(points_cam), min(num_points, len(points_cam)), replace=False)
            points_cam = points_cam[indices]
    
    # Count mask pixels for validation
    mask_pixel_count = np.sum(mask)
    full_image_pixel_count = depth.shape[0] * depth.shape[1]
    mask_coverage = 100.0 * mask_pixel_count / full_image_pixel_count
    
    logger.info(f"  Generated {len(points_cam)} query points from mask (uniform grid sampling)")
    logger.info(f"    Mask coverage: {mask_pixel_count}/{full_image_pixel_count} pixels ({mask_coverage:.1f}%)")
    
    if mask_coverage > 90.0:
        logger.warning(f"    ⚠ WARNING: Mask covers >90% of image! Likely tracking background/floor too!")
    
    logger.info(f"    Camera-space bounds: X=[{points_cam[:,0].min():.2f}, {points_cam[:,0].max():.2f}], "
                f"Y=[{points_cam[:,1].min():.2f}, {points_cam[:,1].max():.2f}], "
                f"Z=[{points_cam[:,2].min():.2f}, {points_cam[:,2].max():.2f}]")
    
    # Keep points in camera space - they will be transformed to world space
    # before being passed to MVTracker (see transform_points_cam_to_world)
    points_3d = points_cam
    
    # Sample points if too many (reduce memory usage)
    max_points = 1000 
    if len(points_3d) > max_points:
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        logger.info(f"  Sampled down to {len(points_3d)} points for memory efficiency")
    
    return points_3d


def transform_points_cam_to_world(points_cam, extrinsic_c2w):
    """
    Transform 3D points from camera space to world space.
    
    Args:
        points_cam: [N, 3] 3D points in camera space [x, y, z]
        extrinsic_c2w: [4, 4] camera-to-world extrinsic matrix
    
    Returns:
        points_world: [N, 3] 3D points in world space
    """
    R = extrinsic_c2w[:3, :3]  # [3, 3] rotation matrix
    t = extrinsic_c2w[:3, 3]   # [3] translation vector
    
    # points_world = R @ points_cam^T + t
    # Reshape for batch matrix multiplication
    points_world = (R @ points_cam.T).T + t  # [N, 3]
    
    return points_world.astype(np.float32)


def load_separated_masks(base_path, case_name, num_cam, video_height, video_width):
    """Load and separate object and controller masks from first frame."""
    target_shape = (video_width, video_height)
    object_masks, controller_masks, mask_info = [], [], {}
    
    for cam_id in range(num_cam):
        # Load metadata
        try:
            with open(f"{base_path}/{case_name}/mask/mask_info_{cam_id}.json") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logger.error(f"✗ Missing mask metadata for camera {cam_id}: mask_info_{cam_id}.json not found!")
            logger.error(f"    Expected: {base_path}/{case_name}/mask/mask_info_{cam_id}.json")
            raise
        except Exception as e:
            logger.error(f"Error loading mask metadata for camera {cam_id}: {e}")
            raise
        
        # Separate object and controller IDs
        object_id, controller_ids = None, []
        for mask_id_str, class_name in metadata.items():
            mask_id = int(mask_id_str)
            if class_name == "hand":
                controller_ids.append(mask_id)
            else:
                object_id = mask_id
        
        # Object and controller IDs identified
        mask_info[cam_id] = {"object_id": object_id, "controller_ids": controller_ids}
        
        # Load object mask - MUST exist and load properly
        object_mask = None
        if object_id is not None:
            mask_path = f"{base_path}/{case_name}/mask/{cam_id}/{object_id}/0.png"
            if os.path.exists(mask_path):
                loaded = read_mask(mask_path)
                if loaded is not None:
                    object_mask = resize_mask(loaded, target_shape)
                    if object_mask is not None:
                        mask_pixels = np.sum(object_mask)
                    else:
                        logger.warning(f"    ⚠ Failed to resize object mask (ID {object_id})")
                else:
                    logger.warning(f"    ⚠ Failed to read object mask (ID {object_id}): {mask_path}")
            else:
                logger.warning(f"    ⚠ Object mask file not found: {mask_path}")
        else:
            logger.warning(f"    ⚠ No object ID found in metadata for camera {cam_id}")
        
        # Fallback: if object mask still missing, use full image but with a WARNING
        if object_mask is None:
            logger.warning(f"    ⚠⚠ USING FULL IMAGE AS FALLBACK - This will track everything!")
            object_mask = np.ones((video_height, video_width), dtype=bool)
        
        # Load and combine controller masks
        controller_mask = np.zeros((video_height, video_width), dtype=bool)
        if controller_ids:
            for ctrl_id in controller_ids:
                mask_path = f"{base_path}/{case_name}/mask/{cam_id}/{ctrl_id}/0.png"
                if os.path.exists(mask_path):
                    loaded = read_mask(mask_path)
                    if loaded is not None:
                        loaded = resize_mask(loaded, target_shape)
                        if loaded is not None:
                            controller_mask = np.logical_or(controller_mask, loaded)
                            mask_pixels = np.sum(loaded)
                        else:
                            logger.warning(f"    ⚠ Failed to resize controller mask (ID {ctrl_id})")
                    else:
                        logger.warning(f"    ⚠ Failed to read controller mask (ID {ctrl_id}): {mask_path}")
                else:
                    logger.warning(f"    ⚠ Controller mask file not found: {mask_path}")
        else:
            logger.warning(f"    ⚠ No controller IDs found in metadata for camera {cam_id}")
        
        # Log overall mask statistics
        obj_pixels = np.sum(object_mask)
        ctrl_pixels = np.sum(controller_mask)
        total_pixels = video_height * video_width
        
        object_masks.append(object_mask)
        controller_masks.append(controller_mask)
    
    logger.info("")
    return object_masks, controller_masks, mask_info


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
