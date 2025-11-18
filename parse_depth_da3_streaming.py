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

import os
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
        """
        Args:
            blend_mode: How to blend overlapping regions ("linear", "avg", or "last")
        """
        self.blend_mode = blend_mode
        self.prev_chunk_depths: Dict[int, List[np.ndarray]] = {}
        self.prev_chunk_overlap_start: int = 0
    
    def align_and_blend(
        self,
        chunk_depths: Dict[int, List[np.ndarray]],  # {cam_id: [depth_maps]}
        chunk_start_idx: int,
        overlap_size: int,
        is_first_chunk: bool = False
    ) -> Dict[int, List[np.ndarray]]:
        """
        Align current chunk with previous chunk in overlapping region.
        
        CRITICAL UNDERSTANDING:
        - First chunk: Return as-is
        - Later chunks: Blend overlap, then return FULL chunk (overlap + new)
        - The save function handles skipping already-saved overlap frames
        
        Args:
            chunk_depths: Depth maps for current chunk, per camera
            chunk_start_idx: Global frame index where this chunk starts
            overlap_size: Number of overlapping frames
            is_first_chunk: Whether this is the first chunk
            
        Returns:
            Aligned depth maps (full chunk with blended overlap)
        """
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
        
        # Update state for next chunk - store the full aligned chunk
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
    
    def reset(self):
        """Reset alignment state."""
        self.prev_chunk_depths.clear()
        self.prev_chunk_overlap_start = 0


# ═══════════════════════════════════════════════════════════════════════════
# Video Writer Manager
# ═══════════════════════════════════════════════════════════════════════════

class VideoWriterManager:
    """Manages video writers for multiple cameras."""
    
    def __init__(self, fps: int = 30, codec: str = "mp4v"):
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writers: Dict[int, cv2.VideoWriter] = {}
        self.output_paths: Dict[int, Path] = {}
    
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
            self.output_paths[camera_id] = output_path
        
        return self.writers[camera_id]
    
    def write_frame(self, camera_id: int, frame: np.ndarray):
        """Write frame to camera's video."""
        if camera_id in self.writers:
            self.writers[camera_id].write(frame)
    
    def release_all(self):
        """Release all video writers."""
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(case_dir: Path) -> Tuple[List[CameraConfig], List[int]]:
    """Load camera calibration data."""
    calib_path = case_dir / "calibrate.pkl"
    with open(calib_path, "rb") as f:
        c2w_list = pickle.load(f)
    
    meta_path = case_dir / "metadata.json"
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


def find_synchronized_frames(camera_configs: List[CameraConfig]) -> List[str]:
    """Find frame filenames that exist across all cameras."""
    frame_sets = []
    for config in camera_configs:
        if not config.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {config.image_dir}")
        
        frames = {f.name for f in config.image_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}}
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
    device: torch.device
) -> Dict[int, List[np.ndarray]]:
    """
    Process a single chunk of frames.
    
    Returns:
        Dictionary mapping camera_id to list of depth maps
    """
    # Collect all images for this chunk
    all_images = []
    all_intrinsics = []
    all_extrinsics = []
    
    for frame_name in frame_names:
        for config in camera_configs:
            img_path = config.image_dir / frame_name
            all_images.append(str(img_path))
            all_intrinsics.append(config.intrinsics)
            all_extrinsics.append(config.extrinsics)
    
    # Stack calibration data
    intrinsics_tensor = np.stack(all_intrinsics, axis=0)
    extrinsics_tensor = np.stack(all_extrinsics, axis=0)
    
    # Run inference on entire chunk
    with torch.no_grad():
        prediction = model.inference(
            image=all_images,
            intrinsics=intrinsics_tensor,
            extrinsics=extrinsics_tensor
        )
    
    # Reorganize results by camera
    num_frames = len(frame_names)
    num_cams = len(camera_ids)
    
    chunk_depths = {cam_id: [] for cam_id in camera_ids}
    
    for frame_idx in range(num_frames):
        for cam_idx, cam_id in enumerate(camera_ids):
            global_idx = frame_idx * num_cams + cam_idx
            
            depth = prediction.depth[global_idx]
            
            if torch.is_tensor(depth):
                depth = depth.detach().cpu().numpy()
            
            chunk_depths[cam_id].append(depth)
    
    return chunk_depths


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
    """Save depth maps and update videos.
    
    CRITICAL: Only saves non-overlapping frames to avoid duplicates!
    - First chunk: saves all frames
    - Later chunks: skips the overlap region (already saved by previous chunk)
    """
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
    args: argparse.Namespace
):
    """
    Main processing loop using chunk-based streaming.
    """
    chunk_config = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    aligner = ChunkAligner(blend_mode=args.blend_mode)
    video_manager = VideoWriterManager(fps=args.fps)
    
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
            
            # Process chunk
            chunk_depths = process_chunk(
                model=model,
                camera_configs=camera_configs,
                camera_ids=camera_ids,
                frame_names=chunk_frames,
                device=args.device
            )
            
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
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("parsed_depth_DA3_streaming_test"))
    
    # Model
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Chunk configuration
    parser.add_argument("--chunk_size", type=int, default=5, 
                       help="Number of frames per chunk")
    parser.add_argument("--overlap", type=int, default=4,
                       help="Overlapping frames between chunks")
    parser.add_argument("--blend_mode", choices=["linear", "avg", "last"], default="linear",
                       help="How to blend overlapping regions")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true")
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
    
    # Load data
    print("\n[1/4] Loading calibration...")
    camera_configs, camera_ids = load_calibration(args.case_dir)
    print(f"      {len(camera_configs)} cameras")
    
    print("\n[2/4] Finding synchronized frames...")
    frame_names = find_synchronized_frames(camera_configs)
    print(f"      {len(frame_names)} frames")
    
    print(f"\n[3/4] Loading model...")
    torch.set_grad_enabled(False)
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()
    
    print("\n[4/4] Processing chunks...")
    process_multi_camera_streaming(
        model=model,
        camera_configs=camera_configs,
        camera_ids=camera_ids,
        frame_names=frame_names,
        output_root=output_dir,
        args=args
    )
    
    print("\n" + "=" * 80)
    print("✓ Complete!")
    print(f"✓ Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()