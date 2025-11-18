#!/usr/bin/env python3
"""
Compare reference depth (from RealSense) with predicted depth (from Video-Depth-Anything).

This script focuses on RELATIVE depth comparison - testing if depth ordering and
ratios between points are preserved, not absolute distances.

Metrics:
- Spearman Rank Correlation: Are points ordered correctly by depth?
- Gradient Correlation: Are local depth changes (edges) preserved?
- Scale-Invariant RMSE: Error after optimal scale+shift alignment
- Ordinal Error: % of point pairs with incorrect depth ordering

Usage:
    python compare_depth.py --case_name double_stretch_sloth
    python compare_depth.py --case_name double_stretch_sloth --camera 0
    python compare_depth.py --case_name double_stretch_sloth --max_frames 50
"""

import argparse
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import stats
from scipy import ndimage


def load_depth_frames(depth_dir, max_frames=-1):
    """Load all depth frames from a directory"""
    depth_dir = Path(depth_dir)
    
    # Get all npy files and sort numerically
    npy_files = list(depth_dir.glob('*.npy'))
    npy_files = sorted(npy_files, key=lambda x: int(x.stem))
    
    if max_frames > 0:
        npy_files = npy_files[:max_frames]
    
    depths = []
    for f in npy_files:
        depth = np.load(f)
        depths.append(depth)
    
    return depths, [f.stem for f in npy_files]


def compute_scale_shift(ref, pred, valid_mask):
    """Compute optimal scale and shift to align pred to ref (least squares)"""
    ref_valid = ref[valid_mask].flatten()
    pred_valid = pred[valid_mask].flatten()
    
    if len(ref_valid) < 2:
        return 1.0, 0.0
    
    # Solve: ref = scale * pred + shift
    A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    result = np.linalg.lstsq(A, ref_valid, rcond=None)
    scale, shift = result[0]
    
    return scale, shift


def compute_relative_metrics(ref_depth, pred_depth, n_sample_pairs=10000):
    """Compute relative depth metrics"""
    
    # Create mask for valid depth values
    valid_mask = (ref_depth > 0) & (pred_depth > 0) & \
                 np.isfinite(ref_depth) & np.isfinite(pred_depth)
    
    if valid_mask.sum() < 100:
        return {
            'spearman': np.nan,
            'gradient_corr': np.nan,
            'si_rmse': np.nan,
            'ordinal_error': np.nan,
            'valid_pixels': 0,
            'scale': 1.0,
            'shift': 0.0,
            'inverted': False
        }
    
    ref = ref_depth.copy()
    pred = pred_depth.copy()
    
    # Check if depth is inverted (negative correlation)
    ref_flat = ref[valid_mask].flatten()
    pred_flat = pred[valid_mask].flatten()
    
    # Quick correlation check
    if len(ref_flat) > 10000:
        idx = np.random.choice(len(ref_flat), 10000, replace=False)
        corr_check, _ = stats.spearmanr(ref_flat[idx], pred_flat[idx])
    else:
        corr_check, _ = stats.spearmanr(ref_flat, pred_flat)
    
    # If negative correlation, invert the prediction
    inverted = corr_check < 0
    if inverted:
        # Invert depth: use 1/depth or max-depth
        pred_max = pred[valid_mask].max()
        pred = pred_max - pred  # Flip the depth
        pred[pred < 0] = 0
    
    # 1. Compute optimal scale and shift
    scale, shift = compute_scale_shift(ref, pred, valid_mask)
    pred_aligned = pred * scale + shift
    
    # 2. Scale-Invariant RMSE (after alignment)
    diff = ref[valid_mask] - pred_aligned[valid_mask]
    si_rmse = np.sqrt(np.mean(diff ** 2))
    
    # 3. Spearman Rank Correlation (depth ordering)
    ref_flat = ref[valid_mask].flatten()
    pred_flat = pred[valid_mask].flatten()
    
    # Subsample for speed if too many pixels
    if len(ref_flat) > 50000:
        idx = np.random.choice(len(ref_flat), 50000, replace=False)
        ref_sample = ref_flat[idx]
        pred_sample = pred_flat[idx]
    else:
        ref_sample = ref_flat
        pred_sample = pred_flat
    
    spearman_corr, _ = stats.spearmanr(ref_sample, pred_sample)
    
    # 4. Gradient Correlation (local structure preservation)
    # Compute depth gradients
    ref_grad_x = ndimage.sobel(ref, axis=1)
    ref_grad_y = ndimage.sobel(ref, axis=0)
    pred_grad_x = ndimage.sobel(pred, axis=1)
    pred_grad_y = ndimage.sobel(pred, axis=0)
    
    ref_grad_mag = np.sqrt(ref_grad_x**2 + ref_grad_y**2)
    pred_grad_mag = np.sqrt(pred_grad_x**2 + pred_grad_y**2)
    
    # Correlation of gradient magnitudes
    grad_valid = valid_mask & (ref_grad_mag > 0.001) & (pred_grad_mag > 0.001)
    if grad_valid.sum() > 100:
        gradient_corr, _ = stats.pearsonr(
            ref_grad_mag[grad_valid].flatten(),
            pred_grad_mag[grad_valid].flatten()
        )
    else:
        gradient_corr = np.nan
    
    # 5. Ordinal Error (% of point pairs with wrong ordering)
    # Sample random pairs
    valid_indices = np.where(valid_mask.flatten())[0]
    if len(valid_indices) > 1000:
        sample_idx = np.random.choice(len(valid_indices), 1000, replace=False)
        valid_indices = valid_indices[sample_idx]
    
    n_pairs = min(n_sample_pairs, len(valid_indices) * (len(valid_indices) - 1) // 2)
    
    if n_pairs > 0:
        # Generate random pairs
        pair_i = np.random.randint(0, len(valid_indices), n_pairs)
        pair_j = np.random.randint(0, len(valid_indices), n_pairs)
        
        # Ensure i != j
        mask = pair_i != pair_j
        pair_i = pair_i[mask]
        pair_j = pair_j[mask]
        
        ref_flat_all = ref.flatten()
        pred_flat_all = pred.flatten()
        
        ref_i = ref_flat_all[valid_indices[pair_i]]
        ref_j = ref_flat_all[valid_indices[pair_j]]
        pred_i = pred_flat_all[valid_indices[pair_i]]
        pred_j = pred_flat_all[valid_indices[pair_j]]
        
        # Check if ordering is preserved
        ref_order = ref_i > ref_j  # True if i is farther
        pred_order = pred_i > pred_j
        
        # Only count pairs with significant depth difference
        significant = np.abs(ref_i - ref_j) > 0.01  # 1cm threshold
        if significant.sum() > 0:
            ordinal_error = (ref_order[significant] != pred_order[significant]).mean()
        else:
            ordinal_error = np.nan
    else:
        ordinal_error = np.nan
    
    return {
        'spearman': spearman_corr,
        'gradient_corr': gradient_corr,
        'si_rmse': si_rmse,
        'ordinal_error': ordinal_error,
        'valid_pixels': valid_mask.sum(),
        'scale': scale,
        'shift': shift,
        'inverted': inverted
    }


def normalize_depth_for_vis(depth, vmin=None, vmax=None):
    """Normalize depth for visualization"""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if len(valid) == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    
    if vmin is None:
        vmin = np.percentile(valid, 2)
    if vmax is None:
        vmax = np.percentile(valid, 98)
    
    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    
    return depth_vis


def draw_text_with_bg(img, text, pos, font, font_scale, text_color, thickness, bg_color=(0, 0, 0), padding=5):
    """Draw text with a background rectangle for better readability"""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Draw background rectangle
    cv2.rectangle(img, 
                  (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + baseline + padding), 
                  bg_color, -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def create_comparison_frame(ref_depth, pred_depth, frame_idx, metrics, color_img=None):
    """Create a comparison frame with optional source RGB
    
    Layout (3 rows):
    - Row 1: Source RGB (full width) with frame info overlay
    - Row 2: Reference Depth | Predicted Depth (aligned)
    - Row 3: Error Map | Metrics Panel
    """
    
    # Handle inverted depth
    pred_proc = pred_depth.copy()
    inverted = metrics.get('inverted', False)
    if inverted:
        valid_mask_temp = (pred_proc > 0) & np.isfinite(pred_proc)
        pred_max = pred_proc[valid_mask_temp].max() if valid_mask_temp.any() else 1.0
        pred_proc = pred_max - pred_proc
        pred_proc[pred_proc < 0] = 0
    
    # Scale predicted depth for visualization
    scale = metrics.get('scale', 1.0)
    shift = metrics.get('shift', 0.0)
    pred_aligned = pred_proc * scale + shift
    
    # Get common depth range for visualization (from reference)
    ref_valid = ref_depth[np.isfinite(ref_depth) & (ref_depth > 0)]
    if len(ref_valid) > 0:
        vmin = np.percentile(ref_valid, 2)
        vmax = np.percentile(ref_valid, 98)
    else:
        vmin, vmax = 0, 1
    
    # Visualize depths (both normalized to same range)
    ref_vis = normalize_depth_for_vis(ref_depth, vmin, vmax)
    pred_vis = normalize_depth_for_vis(pred_aligned, vmin, vmax)
    
    # Compute relative error map (after alignment)
    valid_mask = (ref_depth > 0) & (pred_aligned > 0) & \
                 np.isfinite(ref_depth) & np.isfinite(pred_aligned)
    rel_error = np.zeros_like(ref_depth)
    rel_error[valid_mask] = np.abs(ref_depth[valid_mask] - pred_aligned[valid_mask]) / (ref_depth[valid_mask] + 1e-6)
    
    # Normalize relative error for visualization (0 to 50% range)
    error_norm = np.clip(rel_error / 0.5, 0, 1)
    error_vis = (error_norm * 255).astype(np.uint8)
    error_vis = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)
    error_vis[~valid_mask] = [0, 0, 0]
    
    # Determine target size from reference
    h, w = ref_vis.shape[:2]
    
    # Resize all depth visuals to same size
    if pred_vis.shape[:2] != (h, w):
        pred_vis = cv2.resize(pred_vis, (w, h))
        error_vis = cv2.resize(error_vis, (w, h))
    
    # Font settings for better readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 40  # Taller label bars
    
    # === Row 1: Source RGB ===
    total_width = w * 2
    rgb_row_h = int(h * 0.6)  # Make RGB row 60% height of depth row
    
    if color_img is not None:
        # Resize color image to fit the row while maintaining aspect ratio
        color_h, color_w = color_img.shape[:2]
        scale_factor = min(total_width / color_w, rgb_row_h / color_h)
        new_w = int(color_w * scale_factor)
        new_h = int(color_h * scale_factor)
        color_resized = cv2.resize(color_img, (new_w, new_h))
        
        # Center in row
        rgb_row = np.zeros((rgb_row_h, total_width, 3), dtype=np.uint8)
        x_offset = (total_width - new_w) // 2
        y_offset = (rgb_row_h - new_h) // 2
        rgb_row[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = color_resized
    else:
        # No color image - show placeholder
        rgb_row = np.zeros((rgb_row_h, total_width, 3), dtype=np.uint8)
        cv2.putText(rgb_row, "Source RGB Not Available", 
                    (total_width//2 - 180, rgb_row_h//2), 
                    font, 1.0, (100, 100, 100), 2, cv2.LINE_AA)
    
    # Add frame info overlay on RGB row
    frame_text = f"Frame {frame_idx}"
    draw_text_with_bg(rgb_row, frame_text, (15, 35), font, 1.0, (255, 255, 255), 2, (0, 0, 0, 180))
    
    # Add correlation overlay
    spearman = metrics['spearman']
    corr_color = (0, 255, 0) if spearman > 0.8 else (0, 255, 255) if spearman > 0.5 else (0, 0, 255)
    corr_text = f"Spearman: {spearman:.4f}"
    draw_text_with_bg(rgb_row, corr_text, (total_width - 280, 35), font, 1.0, corr_color, 2)
    
    # Label for RGB row
    rgb_label = np.zeros((label_h, total_width, 3), dtype=np.uint8)
    rgb_label[:] = (40, 40, 40)  # Dark gray background
    cv2.putText(rgb_label, "SOURCE RGB", (total_width//2 - 80, 28), 
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # === Row 2: Depth comparison ===
    depth_row = np.hstack([ref_vis, pred_vis])
    
    # Label for depth row
    depth_label = np.zeros((label_h, total_width, 3), dtype=np.uint8)
    depth_label[:] = (40, 40, 40)
    cv2.putText(depth_label, "REFERENCE (RealSense)", (w//4 - 120, 28), 
                font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(depth_label, "PREDICTED (Aligned)", (w + w//4 - 110, 28), 
                font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # === Row 3: Error map and metrics ===
    # Create metrics panel with larger text
    metrics_panel = np.zeros((h, w, 3), dtype=np.uint8)
    metrics_panel[:] = (30, 30, 30)  # Dark background
    
    y_start = 40
    line_height = 35
    
    # Title
    cv2.putText(metrics_panel, "QUALITY METRICS", (20, y_start), 
                font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    y_start += line_height + 10
    
    # Relative metrics section
    cv2.putText(metrics_panel, "Relative Metrics:", (20, y_start), 
                font, 0.65, (150, 150, 150), 1, cv2.LINE_AA)
    y_start += line_height
    
    metric_items = [
        (f"Spearman Rank: {metrics['spearman']:.4f}", metrics['spearman'] > 0.8),
        (f"Gradient Corr: {metrics['gradient_corr']:.4f}", metrics['gradient_corr'] > 0.8),
        (f"Ordinal Error: {metrics['ordinal_error']*100:.1f}%", metrics['ordinal_error'] < 0.15),
    ]
    
    for text, is_good in metric_items:
        color = (0, 255, 0) if is_good else (0, 200, 255)  # Green if good, yellow otherwise
        cv2.putText(metrics_panel, text, (30, y_start), 
                    font, 0.7, color, 2, cv2.LINE_AA)
        y_start += line_height
    
    y_start += 15
    
    # Aligned metrics section
    cv2.putText(metrics_panel, "After Alignment:", (20, y_start), 
                font, 0.65, (150, 150, 150), 1, cv2.LINE_AA)
    y_start += line_height
    
    aligned_items = [
        f"SI-RMSE: {metrics['si_rmse']:.4f} m",
        f"Scale: {metrics['scale']:.3f}",
        f"Shift: {metrics['shift']:.3f} m",
    ]
    
    for text in aligned_items:
        cv2.putText(metrics_panel, text, (30, y_start), 
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_start += line_height
    
    # Add inverted indicator if applicable
    if inverted:
        y_start += 15
        cv2.putText(metrics_panel, "[Depth Inverted]", (20, y_start), 
                    font, 0.6, (100, 200, 255), 1, cv2.LINE_AA)
    
    error_row = np.hstack([error_vis, metrics_panel])
    
    # Label for error row
    error_label = np.zeros((label_h, total_width, 3), dtype=np.uint8)
    error_label[:] = (40, 40, 40)
    cv2.putText(error_label, "RELATIVE ERROR (%)", (w//4 - 100, 28), 
                font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(error_label, "METRICS", (w + w//4 - 50, 28), 
                font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # === Combine all rows ===
    frame = np.vstack([
        rgb_label, rgb_row,
        depth_label, depth_row,
        error_label, error_row
    ])
    
    return frame


def compare_camera(ref_dir, pred_dir, color_dir, output_dir, camera_id, max_frames=-1):
    """Compare depths for a single camera"""
    
    print(f"\n  Camera {camera_id}:")
    print(f"    Reference: {ref_dir}")
    print(f"    Predicted: {pred_dir}")
    print(f"    Color: {color_dir}")
    
    # Load depths
    ref_depths, ref_names = load_depth_frames(ref_dir, max_frames)
    pred_depths, pred_names = load_depth_frames(pred_dir, max_frames)
    
    # Load color frames
    color_dir = Path(color_dir)
    color_frames = {}
    if color_dir.exists():
        for img_file in color_dir.glob('*.png'):
            frame_name = img_file.stem
            color_frames[frame_name] = cv2.imread(str(img_file))
        for img_file in color_dir.glob('*.jpg'):
            frame_name = img_file.stem
            color_frames[frame_name] = cv2.imread(str(img_file))
    
    if len(ref_depths) == 0:
        print(f"    Error: No reference depth frames found")
        return None
    if len(pred_depths) == 0:
        print(f"    Error: No predicted depth frames found")
        return None
    
    # Match frames by name
    ref_dict = {name: depth for name, depth in zip(ref_names, ref_depths)}
    pred_dict = {name: depth for name, depth in zip(pred_names, pred_depths)}
    
    common_names = sorted(set(ref_names) & set(pred_names), key=int)
    print(f"    Found {len(common_names)} matching frames")
    print(f"    Found {len(color_frames)} color frames")
    
    if len(common_names) == 0:
        print(f"    Error: No matching frames between reference and predicted")
        return None
    
    # Compute metrics for each frame
    all_metrics = []
    frames = []
    
    for name in common_names:
        ref_depth = ref_dict[name]
        pred_depth = pred_dict[name]
        color_img = color_frames.get(name, None)
        
        # Resize if needed
        if ref_depth.shape != pred_depth.shape:
            pred_depth = cv2.resize(pred_depth, (ref_depth.shape[1], ref_depth.shape[0]))
        
        metrics = compute_relative_metrics(ref_depth, pred_depth)
        all_metrics.append(metrics)
        
        # Create visualization frame
        frame = create_comparison_frame(ref_depth, pred_depth, name, metrics, color_img)
        frames.append(frame)
    
    # Check if depth was inverted
    n_inverted = sum(1 for m in all_metrics if m.get('inverted', False))
    was_inverted = n_inverted > len(all_metrics) / 2
    
    # Aggregate metrics
    agg_metrics = {
        'spearman': np.nanmean([m['spearman'] for m in all_metrics]),
        'gradient_corr': np.nanmean([m['gradient_corr'] for m in all_metrics]),
        'si_rmse': np.nanmean([m['si_rmse'] for m in all_metrics]),
        'ordinal_error': np.nanmean([m['ordinal_error'] for m in all_metrics]),
        'scale': np.nanmean([m['scale'] for m in all_metrics]),
        'shift': np.nanmean([m['shift'] for m in all_metrics]),
        'n_frames': len(common_names),
        'inverted': was_inverted
    }
    
    print(f"\n    === RELATIVE DEPTH METRICS ===")
    if was_inverted:
        print(f"    ⚠️  DEPTH WAS INVERTED (auto-corrected)")
    print(f"    Spearman Rank Corr:  {agg_metrics['spearman']:.4f}  (1.0 = perfect ordering)")
    print(f"    Gradient Corr:       {agg_metrics['gradient_corr']:.4f}  (1.0 = same edges)")
    print(f"    Ordinal Error:       {agg_metrics['ordinal_error']*100:.1f}%  (0% = perfect)")
    print(f"    SI-RMSE (aligned):   {agg_metrics['si_rmse']:.4f} m")
    print(f"    Avg Scale factor:    {agg_metrics['scale']:.3f}")
    print(f"    Avg Shift:           {agg_metrics['shift']:.3f}")
    
    # Save comparison video
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = output_dir / f'comparison_camera_{camera_id}.mp4'
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
    
    for frame in frames:
        writer.write(frame)
    writer.release()
    
    print(f"    Saved comparison video: {video_path}")
    
    # Save metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    frame_ids = list(range(len(all_metrics)))
    
    axes[0, 0].plot(frame_ids, [m['spearman'] for m in all_metrics], 'b-')
    axes[0, 0].set_title('Spearman Rank Correlation')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Good (0.9)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(frame_ids, [m['gradient_corr'] for m in all_metrics], 'r-')
    axes[0, 1].set_title('Gradient Correlation')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good (0.8)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(frame_ids, [m['ordinal_error']*100 for m in all_metrics], 'g-')
    axes[1, 0].set_title('Ordinal Error (%)')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylim(0, 50)
    axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Threshold (10%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(frame_ids, [m['si_rmse'] for m in all_metrics], 'purple')
    axes[1, 1].set_title('Scale-Invariant RMSE (m)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].grid(True)
    
    plt.suptitle(f'Relative Depth Comparison - Camera {camera_id}')
    plt.tight_layout()
    
    plot_path = output_dir / f'metrics_camera_{camera_id}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"    Saved metrics plot: {plot_path}")
    
    return agg_metrics


def main():
    parser = argparse.ArgumentParser(description='Compare reference and predicted depth (relative metrics)')
    parser.add_argument('--case_name', type=str, required=True, help='Case name to compare')
    parser.add_argument('--depth_folder', type=str, required=False, default="vda_depth")
    parser.add_argument('--camera', type=int, default=None, help='Specific camera to compare (0, 1, or 2). Default: all')
    parser.add_argument('--max_frames', type=int, default=-1, help='Max frames to compare (-1 = all)')
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent.resolve()
    ref_base = script_dir / 'data' / 'different_types' / args.case_name / 'depth'
    pred_base = Path(__file__).parent.resolve() / args.depth_folder if args.depth_folder else 'data' / 'vda_depth' / args.case_name
    color_base = script_dir / 'data' / 'different_types' / args.case_name / 'color'
    output_dir = script_dir / 'depth_comparison' / args.case_name
    
    print(f"Comparing RELATIVE depths for case: {args.case_name}")
    print(f"Reference base: {ref_base}")
    print(f"Predicted base: {pred_base}")
    
    if not ref_base.exists():
        print(f"Error: Reference depth directory not found: {ref_base}")
        return 1
    if not pred_base.exists():
        print(f"Error: Predicted depth directory not found: {pred_base}")
        print("Run infer_depth.py first to generate predictions")
        return 1
    
    # Determine which cameras to process
    cameras = [args.camera] if args.camera is not None else [0, 1, 2]
    
    all_camera_metrics = {}
    for camera_id in cameras:
        ref_dir = ref_base / str(camera_id)
        pred_dir = pred_base / str(camera_id)
        color_dir = color_base / str(camera_id)
        
        if not ref_dir.exists():
            print(f"\n  Camera {camera_id}: Reference directory not found, skipping")
            continue
        if not pred_dir.exists():
            print(f"\n  Camera {camera_id}: Predicted directory not found, skipping")
            continue
        
        metrics = compare_camera(ref_dir, pred_dir, color_dir, output_dir, camera_id, args.max_frames)
        if metrics:
            all_camera_metrics[camera_id] = metrics
    
    # Print summary
    if all_camera_metrics:
        print(f"\n{'='*60}")
        print("SUMMARY - RELATIVE DEPTH QUALITY")
        print(f"{'='*60}")
        
        avg_spearman = np.mean([m['spearman'] for m in all_camera_metrics.values()])
        avg_gradient = np.mean([m['gradient_corr'] for m in all_camera_metrics.values()])
        avg_ordinal = np.mean([m['ordinal_error'] for m in all_camera_metrics.values()])
        
        print(f"Avg Spearman Rank Corr:  {avg_spearman:.4f}")
        print(f"Avg Gradient Corr:       {avg_gradient:.4f}")
        print(f"Avg Ordinal Error:       {avg_ordinal*100:.1f}%")
        
        # Quality assessment
        print(f"\n--- Quality Assessment ---")
        if avg_spearman > 0.95 and avg_ordinal < 0.05:
            print("✅ Excellent: Depth ordering very well preserved")
        elif avg_spearman > 0.9 and avg_ordinal < 0.1:
            print("✅ Good: Depth ordering well preserved")
        elif avg_spearman > 0.8 and avg_ordinal < 0.2:
            print("⚠️ Fair: Some depth ordering errors")
        else:
            print("❌ Poor: Significant depth ordering errors")
        
        print(f"\nOutput saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
