#!/usr/bin/env python3
"""
Multi-Camera Depth Estimation with DA3 Chunk-Based Streaming

This script implements chunk-based streaming inference for multi-camera sequences,
inspired by DA3-Streaming and VGGT-Long. It processes long sequences in overlapping
chunks to maintain temporal consistency while staying within memory budgets.

Key difference from standard inference:
- STANDARD: Process all frames at once → high memory, better consistency
- CHUNK-BASED: Process overlapping chunks → lower memory, good consistency via alignment
"""

import argparse
import pickle
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.pose_align import align_poses_umeyama


# ═══════════════════════════════════════════════════════════════════════════
# Configuration & Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CameraConfig:
    """Camera calibration data."""
    intrinsics: np.ndarray  # (3, 3)
    extrinsics: np.ndarray  # (4, 4) world-to-camera
    image_dir: Path


@dataclass
class ChunkConfig:
    """Configuration for chunk-based streaming."""
    chunk_size: int = 60  # Number of frames per chunk
    overlap: int = 30  # Overlapping frames between chunks
    
    def __post_init__(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be < chunk_size ({self.chunk_size})")


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def extract_frame_number(filename: str) -> int:
    """Extract numeric frame number from filename."""
    numbers = re.findall(r"\d+", filename)
    if not numbers:
        raise ValueError(f"No frame number found in filename: {filename}")
    return int(numbers[0])


def visualize_depth(
    depth: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentiles: Tuple[float, float] = (2, 98),
    colormap: int = cv2.COLORMAP_TURBO
) -> np.ndarray:
    """Convert depth map to color visualization."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    
    if not np.any(valid):
        h, w = d.shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    
    valid_vals = d[valid]
    if vmin is None or vmax is None:
        lo, hi = np.percentile(valid_vals, percentiles)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    
    if vmax <= vmin:
        vmax = vmin + 1e-6
    
    norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    
    colored = cv2.applyColorMap(gray, colormap)
    colored[~valid] = 0
    
    return colored


def ensure_4x4_matrix(mat: np.ndarray) -> np.ndarray:
    """Convert (3, 4) extrinsics to (4, 4) homogeneous matrix."""
    if mat.shape == (4, 4):
        return mat
    elif mat.shape == (3, 4):
        result = np.eye(4, dtype=mat.dtype)
        result[:3, :4] = mat
        return result
    else:
        raise ValueError(f"Invalid matrix shape: {mat.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# Chunk Alignment
# ═══════════════════════════════════════════════════════════════════════════

class ChunkAligner:
    """
    Aligns overlapping regions between consecutive chunks to maintain
    temporal consistency, similar to VGGT-Long and DA3-Streaming.
    """
    
    def __init__(self, blend_mode: str = "linear"):
        """Initialize aligner with blend mode (linear, avg, or last)."""
        self.blend_mode = blend_mode
        self.prev_chunk_depths: Dict[int, List[np.ndarray]] = {}
    
    def align_and_blend(
        self,
        chunk_depths: Dict[int, List[np.ndarray]],  # {cam_id: [depth_maps]}
        chunk_start_idx: int,
        overlap_size: int,
        is_first_chunk: bool = False
    ) -> Dict[int, List[np.ndarray]]:
        """Blend overlapping region with previous chunk for temporal consistency."""
        if is_first_chunk or not self.prev_chunk_depths:
            # First chunk: no alignment needed, save everything
            self.prev_chunk_depths = {k: v.copy() for k, v in chunk_depths.items()}
            return chunk_depths
        
        aligned_depths = {}
        
        for cam_id, curr_depths in chunk_depths.items():
            if cam_id not in self.prev_chunk_depths:
                # New camera, no alignment possible
                aligned_depths[cam_id] = curr_depths
                continue
            
            prev_depths = self.prev_chunk_depths[cam_id]
            
            # Extract overlapping regions
            # Previous chunk's last 'overlap_size' frames
            prev_overlap = prev_depths[-overlap_size:]
            # Current chunk's first 'overlap_size' frames  
            curr_overlap = curr_depths[:overlap_size]
            
            if len(prev_overlap) != overlap_size or len(curr_overlap) != overlap_size:
                print(f"Warning: Overlap size mismatch for camera {cam_id}")
                print(f"  Expected: {overlap_size}, Got: prev={len(prev_overlap)}, curr={len(curr_overlap)}")
                aligned_depths[cam_id] = curr_depths
                continue
            
            # Blend overlapping frames
            blended_overlap = self._blend_overlap(prev_overlap, curr_overlap)
            
            # Combine: blended overlap + non-overlapping new frames
            aligned = blended_overlap + curr_depths[overlap_size:]
            aligned_depths[cam_id] = aligned
        
        # Update state for next chunk
        self.prev_chunk_depths = {k: v.copy() for k, v in aligned_depths.items()}
        return aligned_depths

    def _blend_overlap(
        self,
        prev_overlap: List[np.ndarray],
        curr_overlap: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Blend overlapping depth maps."""
        n = len(prev_overlap)
        blended = []
        
        for i in range(n):
            if self.blend_mode == "linear":
                # Linear interpolation: gradually transition from prev to curr
                weight_curr = (i + 1) / (n + 1)
                weight_prev = 1 - weight_curr
                blended_depth = weight_prev * prev_overlap[i] + weight_curr * curr_overlap[i]
            elif self.blend_mode == "avg":
                # Simple average
                blended_depth = 0.5 * (prev_overlap[i] + curr_overlap[i])
            elif self.blend_mode == "last":
                # Use current chunk (last wins)
                blended_depth = curr_overlap[i]
            else:
                raise ValueError(f"Unknown blend mode: {self.blend_mode}")
            
            blended.append(blended_depth)

        return blended


# ═══════════════════════════════════════════════════════════════════════════
# Video Writer Manager
# ═══════════════════════════════════════════════════════════════════════════

class VideoWriterManager:
    """Manages video writers for multiple cameras."""

    def __init__(self, fps: int = 30, codec: str = "mp4v"):
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writers: Dict[int, cv2.VideoWriter] = {}
    
    def get_writer(
        self,
        camera_id: int,
        output_path: Path,
        frame_shape: Tuple[int, int]
    ) -> cv2.VideoWriter:
        """Get or create video writer for camera."""
        if camera_id not in self.writers:
            h, w = frame_shape
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            writer = cv2.VideoWriter(
                str(output_path),
                self.codec,
                self.fps,
                (w, h)
            )
            
            if not writer.isOpened():
                raise RuntimeError(f"Failed to create video writer: {output_path}")
            
            self.writers[camera_id] = writer

        return self.writers[camera_id]

    def release_all(self):
        """Release all video writers."""
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()


def reproject_depths_with_sim3(
    depths: Dict[int, np.ndarray],
    intrinsics: Dict[int, np.ndarray],
    extrinsics_inferred: Dict[int, np.ndarray],
    extrinsics_gt: Dict[int, np.ndarray],
    r: np.ndarray,
    t: np.ndarray,
    scale: float
) -> Dict[int, np.ndarray]:
    """
    Reproject depth maps after applying Sim(3) transformation.
    
    1. Backproject depth using inferred extrinsics → 3D world points
    2. Apply Sim(3) transformation (rotation, translation, scale)
    3. Reproject using ground truth extrinsics → corrected depths
    
    Args:
        depths: Dict camera_id → depth map (H, W)
        intrinsics: Dict camera_id → K (3, 3)
        extrinsics_inferred: Dict camera_id → w2c_inferred (4, 4)
        extrinsics_gt: Dict camera_id → w2c_gt (4, 4)
        r: Rotation matrix (3, 3) from Umeyama
        t: Translation vector (3,) from Umeyama
        scale: Scale factor from Umeyama
    
    Returns:
        Dict camera_id → corrected depth maps
    """
    reprojected = {}
    
    for cam_id, depth in depths.items():
        H, W = depth.shape
        depth = depth.astype(np.float32)
        
        K = intrinsics[cam_id]
        w2c_inferred = extrinsics_inferred[cam_id]
        w2c_gt = extrinsics_gt[cam_id]
        c2w_inferred = np.linalg.inv(w2c_inferred)
        
        # Step 1: Backproject to 3D world using inferred extrinsics
        K_inv = np.linalg.inv(K)
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ones = np.ones((H, W))
        uv1 = np.stack([u, v, ones], axis=2)  # (H, W, 3)
        
        ray = np.einsum('ij,hwj->hwi', K_inv, uv1)  # (H, W, 3)
        X_c = ray[..., 0] * depth
        Y_c = ray[..., 1] * depth
        Z_c = ray[..., 2] * depth
        
        P_cam = np.stack([X_c, Y_c, Z_c, np.ones((H, W))], axis=2)  # (H, W, 4)
        P_world_inferred_homo = np.einsum('ij,hwj->hwi', c2w_inferred, P_cam)  # (H, W, 4)
        P_world_inferred = P_world_inferred_homo[..., :3]  # (H, W, 3)
        
        # Step 2: Apply Sim(3) transformation
        # P_corrected = scale * (r @ P_world_inferred) + t
        P_scaled = scale * P_world_inferred
        P_rotated = np.einsum('ij,hwj->hwi', r, P_scaled)  # (H, W, 3)
        P_world_gt = P_rotated + t[None, None, :]  # (H, W, 3)
        
        # Step 3: Reproject using ground truth extrinsics
        P_world_gt_homo = np.concatenate([P_world_gt, np.ones((H, W, 1))], axis=2)  # (H, W, 4)
        P_cam_gt_homo = np.einsum('ij,hwj->hwi', w2c_gt, P_world_gt_homo)  # (H, W, 4)
        P_cam_gt = P_cam_gt_homo[..., :3]  # (H, W, 3)
        
        # Extract depth (Z coordinate in camera frame)
        depth_corrected = P_cam_gt[..., 2]
        
        # Only keep valid depths (positive Z, matches original valid regions)
        valid = np.isfinite(depth) & (depth > 0)
        depth_corrected[~valid] = 0
        
        reprojected[cam_id] = depth_corrected.astype(depth.dtype)
    
    return reprojected


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(case_dir: Path, calib_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> Optional[Tuple[List[CameraConfig], List[int]]]:
    """Load camera calibration from calibrate.pkl and metadata.json files."""
    if calib_path is None:
        calib_path = case_dir / "calibrate.pkl"
    if meta_path is None:
        meta_path = case_dir / "metadata.json"
    
    # Check if files exist
    if not calib_path.exists() or not meta_path.exists():
        return None
    
    with open(calib_path, "rb") as f:
        c2w_list = pickle.load(f)
    
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    num_cams = len(c2w_list)
    camera_ids = list(range(num_cams))
    
    configs = []
    for cam_id in camera_ids:
        intrinsics = np.asarray(metadata["intrinsics"][cam_id], dtype=np.float32)
        
        c2w = np.asarray(c2w_list[cam_id], dtype=np.float32)
        c2w = ensure_4x4_matrix(c2w)
        w2c = np.linalg.inv(c2w)
        
        img_dir = case_dir / "color" / str(cam_id)
        
        configs.append(CameraConfig(
            intrinsics=intrinsics,
            extrinsics=w2c,
            image_dir=img_dir
        ))
    
    return configs, camera_ids


def infer_calibration(
    model: DepthAnything3,
    case_dir: Path,
    frame_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Infer intrinsics and extrinsics from the first frame using DA3 multi-view.

    Runs a single multi-view inference on all cameras at the first frame. This avoids
    invalid extrinsic averaging by extracting poses from one coherent inference pass.
    """
    print("\n[*] Inferring camera calibration from first frame (multi-view)...")

    color_dir = case_dir / "color"
    camera_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    num_cams = len(camera_dirs)

    if not frame_names:
        raise ValueError("No frames found to infer calibration from")
    first_frame = frame_names[0]

    all_images = []
    for cam_dir in camera_dirs:
        img_path = cam_dir / first_frame
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        all_images.append(str(img_path))

    print(f"    {num_cams} cameras for frame: {first_frame}")

    with torch.no_grad():
        prediction = model.inference(image=all_images, use_ray_pose=True)

    if prediction.intrinsics is None or prediction.extrinsics is None:
        raise RuntimeError("DA3 failed to infer camera intrinsics/extrinsics")

    intrinsics_array = prediction.intrinsics  # (num_cams, 3, 3)
    extrinsics_array = np.stack(
        [ensure_4x4_matrix(prediction.extrinsics[i]) for i in range(num_cams)]
    )  # (num_cams, 4, 4)

    # Sanity check: warn on extreme translation magnitudes
    for cam_id in range(num_cams):
        trans_mag = np.linalg.norm(extrinsics_array[cam_id, :3, 3])
        if trans_mag < 0.01 or trans_mag > 100:
            print(f"      ⚠ Camera {cam_id}: extreme translation magnitude={trans_mag:.4f}")

    print(f"    ✓ intrinsics: {intrinsics_array.shape}, extrinsics: {extrinsics_array.shape}")
    return intrinsics_array, extrinsics_array


def save_calibration(
    case_dir: Path,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    calib_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    frame_count: Optional[int] = None,
    fps: int = 30,
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
):
    """Save calibration to calibrate.pkl and metadata.json files."""
    if calib_path is None:
        calib_path = case_dir / "calibrate.pkl"
    if meta_path is None:
        meta_path = case_dir / "metadata.json"

    print(f"\n[*] Saving calibration...")

    # Convert extrinsics (w2c) to c2w for storage
    c2w_list = []
    for w2c in extrinsics:
        w2c = ensure_4x4_matrix(w2c)
        c2w = np.linalg.inv(w2c)
        c2w_list.append(c2w)

    with open(calib_path, "wb") as f:
        pickle.dump(c2w_list, f)
    print(f"    ✓ Saved {calib_path}")

    # Infer frame count and image dimensions from actual depth maps
    if frame_count is None:
        depth_dir = case_dir / "depth" / "0"
        if depth_dir.exists():
            depth_files = [f for f in depth_dir.iterdir() if f.suffix == ".npy"]
            frame_count = len(depth_files)
        else:
            frame_count = 0

    if image_h is None or image_w is None:
        depth_dir = case_dir / "depth" / "0"
        if depth_dir.exists():
            depth_files = sorted([f for f in depth_dir.iterdir() if f.suffix == ".npy"])
            if depth_files:
                depth_map = np.load(depth_files[0])
                image_h = image_h or depth_map.shape[0]
                image_w = image_w or depth_map.shape[1]

    num_cams = len(intrinsics)
    serial_numbers = list(range(num_cams))

    # Save metadata.json — preserve existing fields, update with new
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    metadata.update({
        "intrinsics": [ixt.tolist() for ixt in intrinsics],
        "serial_numbers": serial_numbers,
        "fps": fps,
    })
    
    # Only add WH if we have valid dimensions
    if image_h is not None and image_w is not None:
        metadata["WH"] = [image_w, image_h]
    
    # Only add frame_num if we have a count
    if frame_count is not None and frame_count > 0:
        metadata["frame_num"] = frame_count

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✓ Saved {meta_path}")


def create_camera_configs_from_calibration(
    case_dir: Path,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray
) -> Tuple[List[CameraConfig], List[int]]:
    """Create camera configs from intrinsics and extrinsics arrays."""
    num_cams = len(intrinsics)
    camera_ids = list(range(num_cams))
    
    configs = []
    for cam_id in camera_ids:
        img_dir = case_dir / "color" / str(cam_id)
        
        configs.append(CameraConfig(
            intrinsics=intrinsics[cam_id],
            extrinsics=extrinsics[cam_id],  # Already world-to-camera
            image_dir=img_dir
        ))
    
    return configs, camera_ids


def find_synchronized_frames_from_dir(case_dir: Path) -> List[str]:
    """Find frame filenames synchronized across all camera directories."""
    color_dir = case_dir / "color"
    if not color_dir.exists():
        raise FileNotFoundError(f"Color directory not found: {color_dir}")
    
    # Find all camera directories
    camera_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    if not camera_dirs:
        raise ValueError("No camera directories found in color folder")
    
    # Find common frames across all cameras
    frame_sets = []
    for cam_dir in camera_dirs:
        frames = {f.name for f in cam_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}}
        frame_sets.append(frames)
    
    common_frames = set.intersection(*frame_sets)
    if not common_frames:
        raise ValueError("No synchronized frames found across all cameras")
    
    return sorted(common_frames, key=extract_frame_number)


# ═══════════════════════════════════════════════════════════════════════════
# Main Processing
# ═══════════════════════════════════════════════════════════════════════════

def process_chunk(
    model: DepthAnything3,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
    device: torch.device,
    calib_source: str = "unknown",
    chunk_idx: int = 0
) -> Dict[int, List[np.ndarray]]:
    """
    Run depth inference on entire chunk at once with temporal consistency.
    
    Key insight: Process all frames together to leverage DA3's temporal constraints,
    not frame-by-frame. This is the actual point of chunk-based streaming.
    
    DA3 input shape: (B, N, 3, H, W)
    - B = num_frames in chunk (temporal batch)
    - N = num_cameras per frame (multi-view)
    
    Strategy:
    1. Collect all frames' images for all cameras into one temporal batch
    2. Pass to DA3 with intrinsics only (let it infer extrinsics freely)
    3. Compute global Umeyama alignment across all frames
    4. Apply Sim(3) reprojection (spatial correction)
    
    DA3 then uses both multi-view and temporal constraints simultaneously.
    """
    num_cams = len(camera_ids)
    num_frames = len(frame_names)
    
    print(f"Processing {num_frames} frames × {num_cams} cameras (full temporal batch)")
    
    # Collect all images for the temporal batch
    all_images = []  # Will be (num_frames * num_cams,) flat list
    all_intrinsics = []  # Will be (num_frames, num_cams, 3, 3)
    all_extrinsics_gt = []  # Will be (num_frames, num_cams, 4, 4)
    
    for frame_idx, frame_name in enumerate(frame_names):
        frame_intrinsics = []
        frame_extrinsics_gt = []

        for config in camera_configs:
            all_images.append(str(config.image_dir / frame_name))
            frame_intrinsics.append(config.intrinsics)
            frame_extrinsics_gt.append(config.extrinsics)

        all_intrinsics.append(frame_intrinsics)
        all_extrinsics_gt.append(frame_extrinsics_gt)
    
    # Stack into proper shapes
    intrinsics_batch = np.stack([np.stack(f, axis=0) for f in all_intrinsics], axis=0)  # (num_frames, num_cams, 3, 3)
    extrinsics_gt_batch = np.stack([np.stack([ensure_4x4_matrix(ext) for ext in f], axis=0) for f in all_extrinsics_gt], axis=0)  # (num_frames, num_cams, 4, 4)
    
    # Flatten to match DA3's expected flat image list format
    intrinsics_flat = intrinsics_batch.reshape(num_frames * num_cams, 3, 3)
    
    print(f"  Batch shapes: images={len(all_images)}, intrinsics_flat={intrinsics_flat.shape}, extrinsics_gt={extrinsics_gt_batch.shape}")
    
    # Single inference pass on entire chunk with temporal + multi-view context
    # DA3 will internally reshape based on its architecture
    with torch.no_grad():
        prediction = model.inference(
            image=all_images,
            intrinsics=intrinsics_flat,  # (num_frames * num_cams, 3, 3)
        )
    
    # Extract results
    if prediction.depth is None or prediction.extrinsics is None:
        raise RuntimeError(f"DA3 failed on chunk {chunk_idx}")
    
    # Reshape outputs: (num_frames * num_cams, H, W) → (num_frames, num_cams, H, W)
    depths = prediction.depth
    if torch.is_tensor(depths):
        depths = depths.cpu().numpy()
    
    # Reshape to (num_frames, num_cams, H, W) if needed
    if depths.shape[0] == num_frames * num_cams:
        H, W = depths.shape[-2:]
        depths = depths.reshape(num_frames, num_cams, H, W)
    
    extrinsics_inferred = prediction.extrinsics
    if torch.is_tensor(extrinsics_inferred):
        extrinsics_inferred = extrinsics_inferred.cpu().numpy()
    
    # Ensure 4x4
    if extrinsics_inferred.shape[-2:] == (3, 4):
        ext_4x4 = np.zeros((*extrinsics_inferred.shape[:-2], 4, 4), dtype=extrinsics_inferred.dtype)
        ext_4x4[..., :3, :4] = extrinsics_inferred
        ext_4x4[..., 3, 3] = 1.0
        extrinsics_inferred = ext_4x4
    
    # Reshape to (num_frames, num_cams, 4, 4) if needed
    if extrinsics_inferred.shape[0] == num_frames * num_cams:
        extrinsics_inferred = extrinsics_inferred.reshape(num_frames, num_cams, 4, 4)
    
    # Flatten for global alignment
    extrinsics_inferred_flat = extrinsics_inferred.reshape(-1, 4, 4)
    extrinsics_gt_flat = extrinsics_gt_batch.reshape(-1, 4, 4)
    
    # Compute global Umeyama alignment
    r, t, scale = align_poses_umeyama(extrinsics_gt_flat, extrinsics_inferred_flat)
    print(f"Global alignment: scale={scale:.4f}")
    
    # Build dictionaries for reprojection
    intrinsics_dict = {camera_ids[i]: camera_configs[i].intrinsics for i in range(num_cams)}
    extrinsics_gt_dict = {camera_ids[i]: camera_configs[i].extrinsics for i in range(num_cams)}
    
    # Reproject each frame's depths with Sim(3) correction
    chunk_depths = {cam_id: [] for cam_id in camera_ids}
    
    for frame_idx in range(num_frames):
        frame_extrinsics_inferred = extrinsics_inferred[frame_idx]
        extrinsics_inferred_dict = {camera_ids[i]: frame_extrinsics_inferred[i] for i in range(num_cams)}
        
        frame_depths = {camera_ids[i]: depths[frame_idx, i] for i in range(num_cams)}
        
        # Reproject with Sim(3) transformation
        corrected_depths = reproject_depths_with_sim3(
            depths=frame_depths,
            intrinsics=intrinsics_dict,
            extrinsics_inferred=extrinsics_inferred_dict,
            extrinsics_gt=extrinsics_gt_dict,
            r=r,
            t=t,
            scale=scale
        )
        
        # Store corrected depths
        for cam_id in camera_ids:
            chunk_depths[cam_id].append(corrected_depths[cam_id])
    
    return chunk_depths


def visualize_first_frame_3d(output_root: Path, verbose: bool = False):
    """Visualize first frame depth maps in 3D using Open3D."""
    try:
        from visualize_3d_scene import visualize_depth_scene

        cam_0_dir = output_root / "0"
        if not cam_0_dir.exists():
            if verbose:
                tqdm.write("  ! Camera 0 directory not found")
            return

        depth_files = sorted([f.name for f in cam_0_dir.iterdir() if f.suffix == ".npy"])
        if not depth_files:
            if verbose:
                tqdm.write("  ! No depth files found")
            return

        frame_name = Path(depth_files[0]).stem
        if verbose:
            tqdm.write(f"  → Visualizing frame: {frame_name}")
        
        visualize_depth_scene(
            depth_root=str(output_root),
            calibrate_path=str(output_root / "calibrate.pkl"),
            metadata_path=str(output_root / "metadata.json"),
            frame_name=frame_name,
            depth_scale=None,
            auto_scale=True,
            show_cameras=True,
            frustum_scale=0.05,
            axis_scale=0.01,
            max_depth=100.0
        )
    except Exception as e:
        tqdm.write(f"  ✗ Error visualizing 3D: {e}")


def save_chunk_results(
    chunk_depths: Dict[int, List[np.ndarray]],
    frame_names: List[str],
    output_root: Path,
    video_manager: VideoWriterManager,
    overlap_size: int,
    is_first_chunk: bool,
    visualize: bool = False,
    verbose: bool = False
):
    """Save depth maps and videos, avoiding duplicate overlap frames."""
    # Determine which frames to save
    if is_first_chunk:
        # First chunk: save everything
        start_save_idx = 0
    else:
        # Later chunks: skip overlap (already saved)
        start_save_idx = overlap_size
    
    # Debug output using tqdm.write to avoid interfering with progress bar
    save_frame_names = [frame_names[i] for i in range(start_save_idx, len(frame_names))]
    if verbose and len(save_frame_names) > 0:
        tqdm.write(f"  → Saving {len(save_frame_names)} frames: {save_frame_names[0]} to {save_frame_names[-1]}")
    
    for cam_id, depths in chunk_depths.items():
        save_dir = output_root / str(cam_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Only save from start_save_idx onwards
        for frame_idx in range(start_save_idx, len(depths)):
            depth = depths[frame_idx]
            frame_name = frame_names[frame_idx]
            base_name = Path(frame_name).stem
            
            # Save depth
            np.save(save_dir / f"{base_name}.npy", depth)
            
            # Visualize
            vis = visualize_depth(depth)
            
            # Write to video
            video_path = save_dir / "depth_video.mp4"
            writer = video_manager.get_writer(cam_id, video_path, vis.shape[:2])
            writer.write(vis)
            
            # Display if requested
            if visualize and frame_idx == start_save_idx:  # Show first saved frame
                cv2.imshow(f"Camera {cam_id}", vis)
                cv2.waitKey(1)


def process_multi_camera_streaming(
    model: DepthAnything3,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
    output_root: Path,
    args: argparse.Namespace,
    calib_source: str = "unknown"
):
    """Process video in overlapping chunks with consistent calibration and blending."""
    chunk_config = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    aligner = ChunkAligner(blend_mode=args.blend_mode)
    video_manager = VideoWriterManager(fps=args.fps)
    
    # Store reference calibration from first camera for validation
    ref_intrinsics = camera_configs[0].intrinsics.copy()
    ref_extrinsics = camera_configs[0].extrinsics.copy()
    
    num_frames = len(frame_names)
    stride = chunk_config.chunk_size - chunk_config.overlap
    
    # Calculate number of chunks needed to cover all frames
    if num_frames <= chunk_config.chunk_size:
        num_chunks = 1
    else:
        # After first chunk, each subsequent chunk adds 'stride' new frames
        remaining_frames = num_frames - chunk_config.chunk_size
        num_chunks = 1 + (remaining_frames + stride - 1) // stride  # Ceiling division
    
    print(f"\nProcessing {num_frames} frames in {num_chunks} chunks")
    print(f"Chunk size: {chunk_config.chunk_size}, Overlap: {chunk_config.overlap}, Stride: {stride}")
    print(f"Calibration source: {calib_source} (CONSISTENT across all chunks)")
    print("")  # Blank line before progress bar
    
    try:
        chunk_idx = 0
        start_idx = 0
        
        # Progress bar for chunks
        pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")
        
        while start_idx < num_frames:
            # Calculate chunk boundaries
            end_idx = min(start_idx + chunk_config.chunk_size, num_frames)
            chunk_frames = frame_names[start_idx:end_idx]
            
            if len(chunk_frames) == 0:
                break
            
            # Update progress bar with current chunk info
            pbar.set_postfix({
                'frames': f'{start_idx}-{end_idx-1}',
                'count': len(chunk_frames)
            })
            
            # Process chunk with consistent calibration
            chunk_depths = process_chunk(
                model=model,
                camera_configs=camera_configs,
                camera_ids=camera_ids,
                frame_names=chunk_frames,
                device=args.device,
                calib_source=calib_source,
                chunk_idx=chunk_idx
            )
            
            # Validate calibration consistency (critical for alignment)
            calib_check_fail = False
            if not np.allclose(camera_configs[0].intrinsics, ref_intrinsics, rtol=1e-5):
                tqdm.write(f"  ⚠ WARNING: Intrinsics changed in chunk {chunk_idx}!")
                calib_check_fail = True
            if not np.allclose(camera_configs[0].extrinsics, ref_extrinsics, rtol=1e-5):
                tqdm.write(f"  ⚠ WARNING: Extrinsics changed in chunk {chunk_idx}!")
                calib_check_fail = True
            
            if calib_check_fail and args.verbose:
                tqdm.write(f"     Ref intrinsics shape: {ref_intrinsics.shape}")
                tqdm.write(f"     Current intrinsics shape: {camera_configs[0].intrinsics.shape}")
            
            # Align with previous chunk
            is_first = (chunk_idx == 0)
            
            # For non-first chunks, verify we have enough overlap
            if not is_first and len(chunk_frames) < chunk_config.overlap:
                tqdm.write(f"  Warning: Last chunk has {len(chunk_frames)} frames, less than overlap {chunk_config.overlap}")
                # Treat as continuation without blending
                aligned_depths = chunk_depths
            else:
                aligned_depths = aligner.align_and_blend(
                    chunk_depths=chunk_depths,
                    chunk_start_idx=start_idx,
                    overlap_size=chunk_config.overlap if not is_first else 0,
                    is_first_chunk=is_first
                )
            
            # Save results
            save_chunk_results(
                chunk_depths=aligned_depths,
                frame_names=chunk_frames,
                output_root=output_root,
                video_manager=video_manager,
                overlap_size=chunk_config.overlap if not is_first else 0,
                is_first_chunk=is_first,
                visualize=args.visualize,
                verbose=args.verbose
            )
            
            # Visualize first frame in 3D as soon as it's generated
            if is_first and args.visualize_3d:
                if args.verbose:
                    tqdm.write("  → Visualizing first frame in 3D...")
                visualize_first_frame_3d(output_root=output_root, verbose=args.verbose)
            
            # Move to next chunk
            start_idx += stride
            chunk_idx += 1
            pbar.update(1)
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        pbar.close()
    
    finally:
        video_manager.release_all()
        cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera chunk-based streaming depth estimation with DA3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O
    parser.add_argument("--case_dir", type=Path, required=True,
                       help="Case directory containing color/ subdirectory with images")
    parser.add_argument("--output_root", type=Path, default=Path("parsed_depth_DA3_streaming"),
                       help="Root output directory")
    
    # Calibration (optional)
    parser.add_argument("--calibration", type=Path, default=None,
                       help="Path to calibrate.pkl (optional, defaults to case_dir/calibrate.pkl)")
    parser.add_argument("--metadata", type=Path, default=None,
                       help="Path to metadata.json (optional, defaults to case_dir/metadata.json)")
    
    # Model
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Chunk configuration
    parser.add_argument("--chunk_size", type=int, default=3, 
                       help="Number of frames per chunk")
    parser.add_argument("--overlap", type=int, default=2,
                       help="Overlapping frames between chunks")
    parser.add_argument("--blend_mode", choices=["linear", "avg", "last"], default="linear",
                       help="How to blend overlapping regions")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_3d", action="store_true", help="Visualize first frame depth map in 3D using Open3D")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug information")
    
    args = parser.parse_args()
    
    if not args.case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {args.case_dir}")
    
    case_name = args.case_dir.name
    output_dir = args.output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Multi-Camera DA3 Chunk-Based Streaming")
    print("=" * 80)
    print(f"Case:          {case_name}")
    print(f"Output:        {output_dir}")
    print(f"Model:         {args.model}")
    print(f"Device:        {args.device}")
    print(f"Chunk size:    {args.chunk_size}")
    print(f"Overlap:       {args.overlap}")
    print(f"Blend mode:    {args.blend_mode}")
    print("=" * 80)
    
    # Step 1: Find frames
    print("\n[1/4] Finding synchronized frames...")
    frame_names = find_synchronized_frames_from_dir(args.case_dir)
    print(f"      {len(frame_names)} frames found")

    # Step 2: Load model
    print(f"\n[2/4] Loading model...")
    torch.set_grad_enabled(False)
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()

    # Step 3: Load or infer calibration
    print("\n[3/4] Loading/inferring calibration...")
    calib_result = load_calibration(args.case_dir, args.calibration, args.metadata)
    inferred = False

    if calib_result is not None:
        camera_configs, camera_ids = calib_result
        print(f"      ✓ Loaded calibration ({len(camera_configs)} cameras)")
        calib_source = "loaded"
    else:
        print("      ! No calibration files found — inferring from first frame")
        intrinsics, extrinsics = infer_calibration(model, args.case_dir, frame_names)
        # Save to case_dir for future runs
        save_calibration(args.case_dir, intrinsics, extrinsics, args.calibration, args.metadata, fps=args.fps)
        camera_configs, camera_ids = create_camera_configs_from_calibration(
            args.case_dir, intrinsics, extrinsics
        )
        calib_source = "inferred"
        inferred = True

    # Always copy calibration to output_dir for output tracking
    out_intrinsics = np.stack([cfg.intrinsics for cfg in camera_configs], axis=0)
    out_extrinsics = np.stack([cfg.extrinsics for cfg in camera_configs], axis=0)
    save_calibration(output_dir, out_intrinsics, out_extrinsics, fps=args.fps)

    # Step 4: Process
    print(f"\n[4/4] Processing chunks (calibration: {calib_source})...")
    process_multi_camera_streaming(
        model=model,
        camera_configs=camera_configs,
        camera_ids=camera_ids,
        frame_names=frame_names,
        output_root=output_dir,
        args=args,
        calib_source=calib_source
    )

    print("\n" + "=" * 80)
    print("✓ Complete!")
    print(f"✓ Output:      {output_dir}")
    print(f"✓ Calibration: {output_dir / 'calibrate.pkl'}")
    print(f"✓ Metadata:    {output_dir / 'metadata.json'}")
    if inferred:
        print(f"   (calibration also saved to case_dir for future use)")
    print("=" * 80)


if __name__ == "__main__":
    main()