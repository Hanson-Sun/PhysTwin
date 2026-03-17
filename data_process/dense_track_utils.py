"""
Shared utility functions for dense video tracking (MVTracker and SpaTrackerV2).
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
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def exist_dir(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_mask(mask_path):
    """Load binary mask from image file.
    
    Args:
        mask_path: Path to mask image file
    
    Returns:
        Binary mask as numpy array, or None if load fails
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (mask > 0) if mask is not None else None


def resize_mask(mask, target_shape):
    """Resize mask to target (W, H) if needed.
    
    Args:
        mask: Binary mask as numpy array
        target_shape: (W, H) target shape
    
    Returns:
        Resized mask or original if already correct shape
    """
    if mask is None or (mask.shape[0], mask.shape[1]) == (target_shape[1], target_shape[0]):
        return mask
    return cv2.resize(mask.astype(np.uint8), target_shape, interpolation=cv2.INTER_NEAREST).astype(bool)


def load_video_frames(video_path):
    """Load RGB frames from video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        [T, H, W, 3] array of video frames
    """
    try:
        frames = iio.imread(video_path, plugin="FFMPEG")
        return frames
    except Exception as e:
        logger.error(f"Failed to load video {video_path}: {e}")
        raise


def load_depth_frames(depth_dir, target_shape=None):
    """Load depth maps from directory of .npy files (sorted by frame number).
    
    Args:
        depth_dir: Directory containing depth .npy files
        target_shape: (W, H) to resize depth maps to, if different from source
    
    Returns:
        [T, H, W] depth maps, or None if not found
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
    """Load camera intrinsics and extrinsics from calibration files.
    
    Args:
        base_path: Base path to dataset
        case_name: Case name
    
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


def load_separated_masks(base_path, case_name, num_cam, video_height, video_width):
    """Load and separate object and controller masks from first frame.
    
    Args:
        base_path: Base path to dataset
        case_name: Case name
        num_cam: Number of cameras
        video_height: Video height
        video_width: Video width
    
    Returns:
        object_masks: List of object masks per camera
        controller_masks: List of controller masks per camera
        mask_info: Dictionary with mask metadata
    """
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


def resize_video_and_depth(video, depth, target_height, intrs):
    """Resize video and depth, updating intrinsics accordingly.
    
    Args:
        video: [T, H, W, 3] video frames
        depth: [T, H, W] depth maps
        target_height: Target height (width scales to maintain aspect ratio)
        intrs: [3, 3] intrinsic matrix
    
    Returns:
        video_resized: [T, H', W', 3] resized video
        depth_resized: [T, H', W'] resized depth
        intrs_resized: [3, 3] updated intrinsics
    """
    T, H, W, C = video.shape
    
    # Calculate target width maintaining aspect ratio
    scale = target_height / H
    target_width = int(W * scale)
    
    logger.info(f"  Resizing from {W}x{H} to {target_width}x{target_height} (scale={scale:.3f})")
    
    # Resize video
    video_resized = []
    for frame in video:
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        video_resized.append(resized)
    video_resized = np.stack(video_resized)  # [T, H', W', 3]
    
    # Resize depth
    depth_resized = []
    for d in depth:
        resized = cv2.resize(d, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        depth_resized.append(resized)
    depth_resized = np.stack(depth_resized)  # [T, H', W']
    
    # Update intrinsics
    intrs_resized = intrs.copy()
    intrs_resized[0, 0] *= scale  # fx
    intrs_resized[1, 1] *= scale  # fy
    intrs_resized[0, 2] *= scale  # cx
    intrs_resized[1, 2] *= scale  # cy
    
    return video_resized, depth_resized, intrs_resized
