#!/usr/bin/env python3
"""
Generic entry point for depth streaming.
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

from da3_camera_config import CameraConfig, refine_extrinsics_icp, ensure_4x4_matrix, normalize_extrinsics_scale
from depth_inference_classes import ChunkProcessor, PoseFinder
from DA3PoseFinder import DA3PoseFinder
from DA3ChunkProcessor import DA3ChunkProcessor
from DUSt3RPoseFinder import DUSt3RPoseFinder
from Pi3XPoseFinder import Pi3XPoseFinder
from Pi3XChunkProcessor import Pi3XChunkProcessor

from parse_depth_pi3 import Pi3X

@dataclass
class ChunkConfig:
    """Configuration for chunk-based streaming."""
    chunk_size: int = 60  # Number of frames per chunk
    overlap: int = 30  # Overlapping frames between chunks
    
    def __post_init__(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be < chunk_size ({self.chunk_size})")

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
    
    # Two-tier convention:
    #   case_dir/metadata.json  → K at original image resolution, WH = image dims (model-agnostic)
    #   output_dir/metadata.json → K at depth-model resolution, WH = depth dims (model-specific)
    # The scaling below handles output_dir files; for case_dir files it is a no-op (WH == image dims).
    stored_wh = metadata.get("WH")

    configs = []
    for cam_id in camera_ids:
        K = np.asarray(metadata["intrinsics"][cam_id], dtype=np.float32)

        # Scale K from depth-res up to original image resolution for inference.
        if stored_wh is not None:
            img_dir = case_dir / "color" / str(cam_id)
            sample_imgs = sorted(img_dir.iterdir()) if img_dir.exists() else []
            if sample_imgs:
                orig_img = cv2.imread(str(sample_imgs[0]))
                if orig_img is not None:
                    orig_h, orig_w = orig_img.shape[:2]
                    depth_w, depth_h = int(stored_wh[0]), int(stored_wh[1])
                    if orig_w != depth_w or orig_h != depth_h:
                        K = K.copy()
                        K[0] *= orig_w / depth_w  # scale fx, cx
                        K[1] *= orig_h / depth_h  # scale fy, cy

        c2w = np.asarray(c2w_list[cam_id], dtype=np.float32)
        c2w = ensure_4x4_matrix(c2w)
        w2c = np.linalg.inv(c2w)

        img_dir = case_dir / "color" / str(cam_id)
        configs.append(CameraConfig(
            intrinsics=K,
            extrinsics=w2c,
            image_dir=img_dir
        ))

    return configs, camera_ids


def save_calibration(
    case_dir: Path,
    intrinsics: np.ndarray,  # depth-res K
    extrinsics: np.ndarray,
    calib_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    fps: int = 30,
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
    frame_count: Optional[int] = None,
):
    """Save calibration to calibrate.pkl and metadata.json files.
    For case_dir: pass intrinsics at original image resolution + image_h/w = image dims.
    For output_dir: pass intrinsics at depth-model resolution + image_h/w = depth dims.
    """
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

    if image_h is not None and image_w is not None:
        metadata["WH"] = [image_w, image_h]

    if frame_count is not None:
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
    camera_ids = []
    
    configs = []
    for cam_id in range(num_cams):
        img_dir = case_dir / "color" / str(cam_id)
        if not img_dir.exists():
            continue
        
        camera_ids.append(cam_id)
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


def visualize_first_frame_3d(
    output_root: Path,
    intrinsics: Optional[np.ndarray] = None,  # (N, 3, 3) at depth-map resolution
    extrinsics: Optional[np.ndarray] = None,  # (N, 4, 4) world-to-camera
    verbose: bool = False,
):
    """Visualize first frame depth maps in 3D using Open3D.

    Pass *intrinsics* and *extrinsics* directly (at depth-map resolution) to
    bypass the metadata-file loading and WH rescaling inside visualize_3d_scene.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
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

        # Prefer directly-supplied arrays to avoid file I/O + WH rescaling.
        if intrinsics is not None and extrinsics is not None:
            visualize_depth_scene(
                depth_root=str(output_root),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                frame_name=frame_name,
                depth_scale=None,
                auto_scale=True,
                show_cameras=True,
                frustum_scale=0.05,
                axis_scale=0.01,
                max_depth=100.0
            )
        else:
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
            if visualize and frame_idx == start_save_idx: 
                vis = visualize_depth(depth)
                cv2.imshow(f"Camera {cam_id}", vis)
                cv2.waitKey(1)


def process_multi_camera_streaming(
    chunk_processor: ChunkProcessor,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
    output_root: Path,
    args: argparse.Namespace,
    aligned_intrinsics: Optional[np.ndarray] = None,  # (N,3,3) at depth-map resolution
    aligned_extrinsics: Optional[np.ndarray] = None,  # (N,4,4) world-to-camera
    ):
    """Process video in overlapping chunks with consistent calibration and blending."""
    chunk_config = ChunkConfig(chunk_size=args.chunk_size,overlap=args.overlap)
    aligner = ChunkAligner(blend_mode=args.blend_mode)
    
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
    print("")  # Blank line before progress bar
    
    try:
        chunk_idx = 0
        start_idx = 0
        pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")
        
        while start_idx < num_frames:
            end_idx = min(start_idx + chunk_config.chunk_size, num_frames)
            chunk_frames = frame_names[start_idx:end_idx]
            
            if len(chunk_frames) == 0:
                break
            
            pbar.set_postfix({'frames': f'{start_idx}-{end_idx-1}', 'count': len(chunk_frames)})
            
            # Process chunk with consistent calibration
            chunk_depths = chunk_processor.process_chunk(
                camera_configs=camera_configs,
                camera_ids=camera_ids,
                frame_names=chunk_frames
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
                overlap_size=chunk_config.overlap if not is_first else 0,
                is_first_chunk=is_first,
                visualize=args.visualize,
                verbose=args.verbose
            )
            
            # Visualize first frame in 3D as soon as it's generated
            if is_first and args.visualize_3d:
                if args.verbose:
                    tqdm.write("  → Visualizing first frame in 3D...")
                visualize_first_frame_3d(
                    output_root=output_root,
                    intrinsics=aligned_intrinsics,
                    extrinsics=aligned_extrinsics,
                    verbose=args.verbose,
                )
            

            start_idx += stride
            chunk_idx += 1
            pbar.update(1)
            
            # Minimal cleanup: essential GPU cache clear only (~1-2ms overhead)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        pbar.close()
    
    finally:
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Chunk configuration
    parser.add_argument("--chunk_size", type=int, default=3, 
                       help="Number of frames per chunk")
    parser.add_argument("--overlap", type=int, default=2,
                       help="Overlapping frames between chunks")
    parser.add_argument("--blend_mode", choices=["linear", "avg", "last"], default="linear",
                       help="How to blend overlapping regions")
    parser.add_argument("--calib_frames", type=int, default=5,
                       help="Number of evenly-spaced frames to average when inferring calibration "
                            "(only used when no calibrate.pkl is found)")
    parser.add_argument("--camera_baseline", type=float, default=1.0,
                       help="Target metric distance (metres) between the two furthest cameras "
                            "when calibration is inferred via DUSt3R. Has no effect when a "
                            "calibrate.pkl is already present.")
    parser.add_argument("--model", default="DA3", help="Depth model: DA3, Pi3X")
    parser.add_argument("--pose_calibration_model", default="DUSt3R", help="DUSt3R, DA3, Pi3X")
    
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
    
    print("\n[1/5] Finding synchronized frames...")
    frame_names = find_synchronized_frames_from_dir(args.case_dir)
    print(f"      {len(frame_names)} frames found")

    print(f"\n[2/5] Loading model {args.model}...")
    if args.model == "DA3":
        depth_model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(args.device).eval()
        depth_pose_finder = DA3PoseFinder(depth_model)
        chunk_processor = DA3ChunkProcessor(depth_model)
    elif args.model == "Pi3X":
        depth_model = Pi3X.from_pretrained("yyfz233/Pi3X").to(args.device).eval()
        depth_pose_finder = Pi3XPoseFinder(depth_model)
        chunk_processor = Pi3XChunkProcessor(depth_model)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


    print("\n[3/5] Loading/inferring calibration...")
    calib_result = load_calibration(args.case_dir, args.calibration, args.metadata)
    inferred = False
    
    if calib_result is not None:
        camera_configs, camera_ids = calib_result
        print(f"      ✓ Loaded calibration ({len(camera_configs)} cameras)")
    else:
        print(f"      ! No calibration files found — inferring from frames using {args.pose_calibration_model}...")
        
        # Only instantiate calib model if needed, use it, then free immediately
        if args.pose_calibration_model == "DUSt3R":
            calib_pose_finder = DUSt3RPoseFinder()  # loads DUSt3R lazily
        elif args.pose_calibration_model == "DA3":
            calib_pose_finder = depth_pose_finder  # reuse depth model (no extra memory)
        elif args.pose_calibration_model == "Pi3X":
            calib_pose_finder = depth_pose_finder  # delegates to DUSt3R internally
        else:
            raise ValueError(f"Unsupported pose_calibration_model: {args.pose_calibration_model}")

        # Infer calibration
        intrinsics, extrinsics = calib_pose_finder.infer_calibration(args.case_dir, frame_names, args.calib_frames)
        
        # Free DUSt3R immediately if we loaded it separately
        if args.pose_calibration_model == "DUSt3R":
            del calib_pose_finder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        extrinsics, _scale = normalize_extrinsics_scale(extrinsics, args.camera_baseline)
        camera_configs, camera_ids = create_camera_configs_from_calibration(
            args.case_dir, intrinsics, extrinsics
        )
        inferred = True

    print("\n[4/5] Determining depth-model output resolution...")

    out_extrinsics = np.stack([cfg.extrinsics for cfg in camera_configs], axis=0)

    # Note: both DA3ChunkProcessor and Pi3XChunkProcessor read image-res K from
    # camera_configs and do their own internal rescaling; they do NOT need this.
    aligned_intrinsics, depth_hw = depth_pose_finder.align_intrinsics(
        camera_configs, camera_ids, frame_names
    )
    if aligned_intrinsics is None:
        print("      ! Model returned no intrinsics — using originals")
        raise RuntimeError("No intrinsics")

    if depth_hw:
        print(f"      ✓ Depth-map output resolution: {depth_hw[1]}×{depth_hw[0]}")
        print(f"      ✓ K for visualizer: fx {camera_configs[0].intrinsics[0,0]:.1f} (image-res) "
              f"→ fx {aligned_intrinsics[0,0,0]:.1f} (depth-res)")


    orig_intrinsics = np.stack([cfg.intrinsics for cfg in camera_configs], axis=0)
    sample_img = cv2.imread(str(camera_configs[0].image_dir / frame_names[0]))
    img_h, img_w = sample_img.shape[:2] if sample_img is not None else (None, None)

    if inferred:
        save_calibration(
            args.case_dir, orig_intrinsics, out_extrinsics, fps=args.fps,
            image_h=img_h,
            image_w=img_w,
            frame_count=len(frame_names),
        )

    save_calibration(
        output_dir, orig_intrinsics, out_extrinsics, fps=args.fps,
        image_h=img_h,
        image_w=img_w,
        frame_count=len(frame_names),
    )
        

    print(f"\n[5/5] Processing chunks...")
    process_multi_camera_streaming(
        chunk_processor=chunk_processor,
        camera_configs=camera_configs,
        camera_ids=camera_ids,
        frame_names=frame_names,
        output_root=output_dir,
        args=args,
        aligned_intrinsics=aligned_intrinsics,
        aligned_extrinsics=out_extrinsics,
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