"""Evaluation script for temporal depth smoothing."""

import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from .inference import DepthSmoother
from .utils import compute_temporal_variance, compute_background_mask
from .losses import compute_motion_mask


def evaluate_smoothing(depth_raw: np.ndarray,
                      depth_smooth: np.ndarray,
                      rgb: np.ndarray,
                      bg_threshold: float = 0.01,
                      motion_threshold: float = 0.05) -> dict:
    """
    Evaluate smoothing quality on static vs. moving regions.
    
    Args:
        depth_raw: [T, H, W] raw depth input
        depth_smooth: [T, H, W] smoothed output
        rgb: [T, H, W, 3] RGB frames
        bg_threshold: background mask threshold
        motion_threshold: motion detection threshold
    
    Returns:
        metrics: dict of evaluation metrics
    """
    depth_raw_t = torch.from_numpy(depth_raw).float()
    depth_smooth_t = torch.from_numpy(depth_smooth).float()
    rgb_t = torch.from_numpy(rgb).float()
    
    metrics = {}
    
    # Background mask (static regions)
    bg_mask = compute_background_mask(depth_raw_t, threshold=bg_threshold)
    
    # Motion mask
    motion_mask = compute_motion_mask(rgb_t, threshold=motion_threshold)
    
    # --- Temporal Variance (Lower = Better) ---
    var_raw = compute_temporal_variance(depth_raw_t)
    var_smooth = compute_temporal_variance(depth_smooth_t)
    
    # On background
    bg_idx = (bg_mask > 0.5).nonzero(as_tuple=True)
    if len(bg_idx[0]) > 0:
        var_raw_bg = var_raw[bg_idx].mean().item()
        var_smooth_bg = var_smooth[bg_idx].mean().item()
        var_reduction = (var_raw_bg - var_smooth_bg) / (var_raw_bg + 1e-6)
        
        metrics['temporal_variance_raw_bg'] = var_raw_bg
        metrics['temporal_variance_smooth_bg'] = var_smooth_bg
        metrics['temporal_variance_reduction_bg'] = var_reduction
    
    # --- Mean Absolute Error from Input (Lower = Better) ---
    mae = torch.abs(depth_smooth_t - depth_raw_t).mean().item()
    
    # On background only (should be minimal)
    if len(bg_idx[0]) > 0:
        mae_bg = torch.abs(depth_smooth_t[:, bg_idx[0], bg_idx[1]] - 
                          depth_raw_t[:, bg_idx[0], bg_idx[1]]).mean().item()
        metrics['mae_bg'] = mae_bg
    
    metrics['mae'] = mae
    
    # --- Gradient Preservation on Moving Regions (Higher = Better) ---
    grad_raw = torch.abs(depth_raw_t[1:] - depth_raw_t[:-1])
    grad_smooth = torch.abs(depth_smooth_t[1:] - depth_smooth_t[:-1])
    
    if motion_mask.sum() > 0:
        motion_idx = (motion_mask > 0.5).nonzero(as_tuple=True)
        grad_corr = torch.nn.functional.cosine_similarity(
            grad_smooth[motion_idx].view(-1),
            grad_raw[motion_idx].view(-1),
            dim=0
        ).item()
        metrics['gradient_correlation_moving'] = grad_corr
    
    # --- Standard Deviation on Background (Should be Minimal) ---
    std_raw_bg = var_raw_bg ** 0.5 if 'var_raw_bg' in locals() else None
    std_smooth_bg = var_smooth_bg ** 0.5 if 'var_smooth_bg' in locals() else None
    
    if std_raw_bg is not None:
        metrics['std_raw_bg'] = std_raw_bg
        metrics['std_smooth_bg'] = std_smooth_bg
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<50} {value:>10.6f}")
    
    print("="*60)
    
    # Interpretation
    print("\nInterpretation:")
    if 'temporal_variance_reduction_bg' in metrics:
        reduction = metrics['temporal_variance_reduction_bg']
        if reduction > 0.5:
            print(f"✓ Excellent temporal smoothing: {100*reduction:.1f}% variance reduction")
        elif reduction > 0.3:
            print(f"✓ Good temporal smoothing: {100*reduction:.1f}% variance reduction")
        else:
            print(f"⚠ Insufficient temporal smoothing: {100*reduction:.1f}% variance reduction")
    
    if 'mae_bg' in metrics:
        mae = metrics['mae_bg']
        print(f"✓ Geometric fidelity (MAE on background): {mae:.6f}")
        if mae > 0.01:
            print(f"  (Consider increasing lambda_geometric if drift is too high)")
    
    if 'gradient_correlation_moving' in metrics:
        corr = metrics['gradient_correlation_moving']
        if corr > 0.95:
            print(f"✓ Motion preservation excellent: {corr:.4f} correlation")
        elif corr > 0.85:
            print(f"✓ Motion preservation good: {corr:.4f} correlation")
        else:
            print(f"⚠ Motion preservation could be improved: {corr:.4f} correlation")


def visualize_results(depth_raw: np.ndarray,
                     depth_smooth: np.ndarray,
                     rgb: np.ndarray,
                     output_path: str = 'evaluation.png') -> None:
    """Visualize before/after smoothing."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Frame indices for visualization
    t_frames = [0, depth_raw.shape[0] // 2, depth_raw.shape[0] - 1]
    
    for col, t in enumerate(t_frames):
        # Raw
        im0 = axes[0, col].imshow(depth_raw[t], cmap='viridis')
        axes[0, col].set_title(f'Raw (frame {t})')
        plt.colorbar(im0, ax=axes[0, col])
        
        # Smoothed
        im1 = axes[1, col].imshow(depth_smooth[t], cmap='viridis')
        axes[1, col].set_title(f'Smoothed (frame {t})')
        plt.colorbar(im1, ax=axes[1, col])
        
        # Difference (magnitude of change)
        diff = np.abs(depth_smooth[t] - depth_raw[t])
        im2 = axes[2, col].imshow(diff, cmap='hot')
        axes[2, col].set_title(f'|Difference| (frame {t})')
        plt.colorbar(im2, ax=axes[2, col])
    
    axes[0, 0].set_ylabel('Raw Depth', fontsize=12)
    axes[1, 0].set_ylabel('Smoothed Depth', fontsize=12)
    axes[2, 0].set_ylabel('Absolute Difference', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate temporal depth smoothing'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Trained model checkpoint')
    parser.add_argument('--depth-input', type=str, required=True,
                       help='Raw depth input (.npy)')
    parser.add_argument('--rgb-input', type=str, required=True,
                       help='RGB frames (.npy)')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                       help='Output directory for visualizations')
    parser.add_argument('--save-smoothed', action='store_true',
                       help='Save smoothed depth output')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    depth_raw = np.load(args.depth_input)
    rgb = np.load(args.rgb_input)
    print(f"  Depth: {depth_raw.shape}")
    print(f"  RGB: {rgb.shape}")
    
    # Normalize RGB
    if rgb.max() > 1:
        rgb = rgb / 255.0
    
    # Inference
    print("\nRunning inference...")
    smoother = DepthSmoother(args.checkpoint, device='cuda')
    depth_smooth = smoother.smooth(depth_raw, rgb)
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_smoothing(depth_raw, depth_smooth, rgb)
    print_metrics(metrics)
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        # Convert np types for JSON serialization
        for key in metrics:
            if isinstance(metrics[key], np.floating):
                metrics[key] = float(metrics[key])
        json.dump(metrics, f, indent=2)
    
    # Save smoothed depth if requested
    if args.save_smoothed:
        output_path = output_dir / 'depth_smooth.npy'
        np.save(output_path, depth_smooth)
        print(f"\nSaved smoothed depth to {output_path}")
    
    # Visualize if requested
    if args.visualize:
        viz_path = output_dir / 'visualization.png'
        visualize_results(depth_raw, depth_smooth, rgb, str(viz_path))


if __name__ == '__main__':
    main()
