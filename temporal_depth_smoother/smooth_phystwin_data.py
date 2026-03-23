"""
Smooth depth sequences from PhysTwin multi-camera capture for a single case.

Loads depth and color frames from individual files, smooths each camera's
depth sequence using the trained model, and saves outputs in both:
  1. Joined arrays (npy) for evaluation/analysis
  2. Original folder structure (individual npy per frame)
"""

import numpy as np
import argparse
import json
import traceback
from pathlib import Path
import cv2

from .inference import DepthSmoother
from .evaluate import evaluate_smoothing, print_metrics


def load_frame(path: Path) -> np.ndarray:
    """Load and convert PNG to RGB.
    
    Args:
        path: Path to PNG file
    
    Returns:
        [H, W, 3] uint8 RGB array
    """
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to load {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth_sequence(depth_dir: Path) -> np.ndarray:
    """Load depth frames from individual .npy files.
    
    Args:
        depth_dir: Path to camera depth folder (e.g., data/different_types/{case}/depth/0/)
    
    Returns:
        [T, H, W] concatenated depth array
    """
    npy_files = sorted(depth_dir.glob('*.npy'), key=lambda p: int(p.stem))
    frames = [np.load(f) for f in npy_files]
    return np.stack(frames, axis=0)  # [T, H, W]


def load_rgb_sequence(rgb_dir: Path) -> np.ndarray:
    """Load RGB frames from individual .png files.
    
    Args:
        rgb_dir: Path to camera RGB folder (e.g., data/different_types/{case}/color/0/)
    
    Returns:
        [T, H, W, 3] uint8 concatenated RGB array
    """
    png_files = sorted(rgb_dir.glob('*.png'), key=lambda p: int(p.stem))
    frames = [load_frame(f) for f in png_files]
    return np.stack(frames, axis=0).astype(np.uint8)  # [T, H, W, 3]


def save_depth_sequence(depth: np.ndarray, output_dir: Path) -> None:
    """Save depth frames as individual .npy files.
    
    Args:
        depth: [T, H, W] depth array
        output_dir: Path to camera depth folder to save to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for t, frame in enumerate(depth):
        np.save(output_dir / f'{t}.npy', frame)


def smooth_camera(camera_id: int,
                  case_dir: Path,
                  smoother: DepthSmoother,
                  output_dir: Path,
                  window_size: int = 16,
                  overlap: int = 4,
                  evaluate: bool = False) -> bool:
    """Smooth depth sequence for a single camera.
    
    Args:
        camera_id: Camera index (0, 1, 2, ...)
        case_dir: Path to case directory (e.g., data/different_types/{case}/)
        smoother: DepthSmoother instance
        output_dir: Path to save outputs
        window_size: Chunk size for inference (frames)
        overlap: Edge frames discarded per chunk
        evaluate: Whether to compute metrics
    
    Returns:
        True if successful, False otherwise
    """
    # Try depth first, then fall back to depth_old (in case it was backed up)
    depth_dir = case_dir / 'depth' / str(camera_id)
    if not depth_dir.exists():
        depth_dir = case_dir / 'depth_old' / str(camera_id)
    
    rgb_dir = case_dir / 'color' / str(camera_id)
    
    if not depth_dir.exists() or not rgb_dir.exists():
        print(f"  Camera {camera_id}: missing depth or RGB dir, skipping", flush=True)
        return True  # Not an error, just skip
    
    print(f"  Camera {camera_id}: loading data...", flush=True)
    depth_raw = load_depth_sequence(depth_dir)
    rgb = load_rgb_sequence(rgb_dir)
    

    if rgb.shape[0] != depth_raw.shape[0]:
        raise RuntimeError(f"RGB and depth frame count mismatch for camera {camera_id}: {rgb.shape[0]} RGB frames vs {depth_raw.shape[0]} depth frames")
    
    # Resize RGB to match depth spatial dimensions if needed
    if rgb.shape[1:3] != depth_raw.shape[1:3]:
        print(f"    Resizing RGB from {rgb.shape[1:3]} to {depth_raw.shape[1:3]}", flush=True)
        target_h, target_w = depth_raw.shape[1:3]
        rgb_resized = np.zeros((rgb.shape[0], target_h, target_w, 3), dtype=np.uint8)
        for t in range(rgb.shape[0]):
            rgb_resized[t] = cv2.resize(rgb[t], (target_w, target_h))
        rgb = rgb_resized
    
    print(f"    Depth: {depth_raw.shape}  RGB: {rgb.shape}", flush=True)
    
    print(f"  Camera {camera_id}: smoothing...", flush=True)
    depth_smooth = smoother.smooth(depth_raw, rgb,
                                   window_size=window_size,
                                   overlap=overlap)
    
    # Save joined arrays for analysis
    cam_output_dir = output_dir / 'joined' / str(camera_id)
    cam_output_dir.mkdir(parents=True, exist_ok=True)
    
    joined_path = cam_output_dir / 'depth_smooth.npy'
    np.save(joined_path, depth_smooth)
    print(f"  Camera {camera_id}: saved joined depth -> {joined_path}", flush=True)
    
    # Save individual frame files in original structure
    frame_output_dir = output_dir / 'frames' / str(camera_id)
    save_depth_sequence(depth_smooth, frame_output_dir)
    print(f"  Camera {camera_id}: saved individual frames -> {frame_output_dir}", flush=True)
    
    # Evaluate if requested
    if evaluate:
        print(f"  Camera {camera_id}: evaluating...", flush=True)
        metrics = evaluate_smoothing(depth_raw, depth_smooth, rgb)
        metrics_path = cam_output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in metrics.items()},
                f, indent=2
            )
        print_metrics(metrics)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Smooth depth sequences from PhysTwin multi-camera data for a single case'
    )
    parser.add_argument('--base-path', default='data/different_types',
                        help='Base directory containing cases')
    parser.add_argument('--case-name', required=True,
                        help='Case name (subdirectory in base_path)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--camera-ids', nargs='+', type=int, default=[0, 1, 2],
                        help='Camera IDs to process')
    parser.add_argument('--window-size', type=int, default=32,
                        help='Chunk size for inference (frames)')
    parser.add_argument('--overlap', type=int, default=4,
                        help='Edge frames discarded per chunk')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Compute evaluation metrics')
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(args.base_path)
    case_dir = base_path / args.case_name
    output_dir = case_dir / 'smoothed'
    
    if not case_dir.exists():
        print(f"Error: case directory not found: {case_dir}")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Smoothing: {args.case_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    smoother = DepthSmoother(args.checkpoint, device=args.device)
    
    # Process each camera
    try:
        for cam_id in args.camera_ids:
            if not smooth_camera(
                cam_id, case_dir, smoother, output_dir,
                window_size=args.window_size,
                overlap=args.overlap,
                evaluate=args.evaluate
            ):
                print(f"\nError: failed to process camera {cam_id}", flush=True)
                return 1
        
        print(f"\n{'='*60}")
        print(f"Success: All cameras processed")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
