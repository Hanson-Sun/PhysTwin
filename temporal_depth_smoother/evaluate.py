"""Evaluation script for temporal depth smoothing."""

import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from .inference import DepthSmoother
from .utils import compute_temporal_variance
from .losses import compute_motion_mask, compute_flicker_mask


def evaluate_smoothing(depth_raw: np.ndarray,
                       depth_smooth: np.ndarray,
                       rgb: np.ndarray,
                       depth_vda: np.ndarray = None,
                       motion_threshold: float = 0.05,
                       depth_threshold: float = 0.01,
                       vda_weight: float = 0.7) -> dict:
    """
    Evaluate smoothing quality on static vs. moving regions.

    Static regions are identified via the flicker mask (VDA + RGB),
    not the background mask — more reliable since it doesn't depend
    on depth temporal variance thresholds.

    Args:
        depth_raw:        [T, H, W] raw depth input (quantile normalised)
        depth_smooth:     [T, H, W] smoothed output
        rgb:              [T, H, W, 3] float [0, 1]
        depth_vda:        [T, H, W] aligned VDA depth (optional)
        motion_threshold: RGB motion detection threshold
        depth_threshold:  normalised depth change threshold for flicker detection
        vda_weight:       VDA vs RGB weight in flicker mask

    Returns:
        metrics dict
    """
    depth_raw_t    = torch.from_numpy(depth_raw).float()
    depth_smooth_t = torch.from_numpy(depth_smooth).float()
    rgb_t          = torch.from_numpy(rgb).float()

    # Add batch dim for loss functions that expect [B, T, H, W]
    raw_b    = depth_raw_t.unsqueeze(0)   # [1, T, H, W]
    smooth_b = depth_smooth_t.unsqueeze(0)
    rgb_b    = rgb_t.unsqueeze(0)         # [1, T, H, W, 3]

    metrics = {}

    # ── Temporal gradients (used throughout) ─────────────────────────────
    grad_raw    = torch.abs(depth_raw_t[1:]    - depth_raw_t[:-1])   # [T-1, H, W]
    grad_smooth = torch.abs(depth_smooth_t[1:] - depth_smooth_t[:-1])

    # ── Flicker mask — primary static region detector ─────────────────────
    if depth_vda is not None:
        vda_b        = torch.from_numpy(depth_vda).float().unsqueeze(0)
        flicker_mask = compute_flicker_mask(
            raw_b, vda_b, rgb_b,
            depth_threshold=depth_threshold,
            rgb_threshold=motion_threshold,
            vda_weight=vda_weight,
        ).squeeze(0)  # [T-1, H, W]
    else:
        # Fall back to RGB-only static detection if no VDA provided
        flicker_mask = (1.0 - compute_motion_mask(
            rgb_b, threshold=motion_threshold
        ).squeeze(0))  # [T-1, H, W], 1=static

    flicker_pixels = flicker_mask > 0.5

    # ── Temporal variance on static pixels ───────────────────────────────
    var_raw    = compute_temporal_variance(depth_raw_t)    # [H, W]
    var_smooth = compute_temporal_variance(depth_smooth_t)

    # Use flicker mask collapsed to spatial — pixel is "static" if it's
    # static in the majority of frame pairs
    static_spatial = (flicker_mask.mean(dim=0) > 0.5)  # [H, W]
    static_idx     = static_spatial.nonzero(as_tuple=True)

    if len(static_idx[0]) > 0:
        var_raw_static    = var_raw[static_idx].mean().item()
        var_smooth_static = var_smooth[static_idx].mean().item()
        var_reduction     = (var_raw_static - var_smooth_static) / (var_raw_static + 1e-6)

        metrics['temporal_variance_raw_static']       = var_raw_static
        metrics['temporal_variance_smooth_static']    = var_smooth_static
        metrics['temporal_variance_reduction_static'] = var_reduction
        metrics['std_raw_static']                     = var_raw_static    ** 0.5
        metrics['std_smooth_static']                  = var_smooth_static ** 0.5

    # ── Flicker change on flagged pixels ─────────────────────────────────
    if flicker_pixels.sum() > 0:
        original_flicker  = grad_raw[flicker_pixels].mean().item()
        residual_flicker  = grad_smooth[flicker_pixels].mean().item()
        flicker_reduction = (original_flicker - residual_flicker) / (original_flicker + 1e-6)

        metrics['flicker_change_raw']     = original_flicker
        metrics['flicker_change_smooth']  = residual_flicker
        metrics['flicker_reduction']      = flicker_reduction
        metrics['flicker_pixel_fraction'] = flicker_pixels.float().mean().item()

    # ── MAE from input (overall and on static pixels) ────────────────────
    metrics['mae'] = torch.abs(depth_smooth_t - depth_raw_t).mean().item()

    if len(static_idx[0]) > 0:
        metrics['mae_static'] = torch.abs(
            depth_smooth_t[:, static_idx[0], static_idx[1]] -
            depth_raw_t[:,   static_idx[0], static_idx[1]]
        ).mean().item()

    # ── Spatial gradient preservation (geometric fidelity) ───────────────
    # Are the spatial depth relationships (edges, relative depths) preserved?
    from .losses import compute_depth_spatial_gradient
    smooth_gx, smooth_gy = compute_depth_spatial_gradient(smooth_b)
    raw_gx,    raw_gy    = compute_depth_spatial_gradient(raw_b)
    spatial_grad_error = ((smooth_gx - raw_gx).abs().mean() +
                          (smooth_gy - raw_gy).abs().mean()) / 2.0
    metrics['spatial_gradient_error'] = spatial_grad_error.item()

    # ── Motion preservation on moving pixels ─────────────────────────────
    motion_mask = compute_motion_mask(rgb_b, threshold=motion_threshold).squeeze(0)
    if motion_mask.sum() > 0:
        motion_idx = (motion_mask > 0.5).nonzero(as_tuple=True)
        grad_corr  = torch.nn.functional.cosine_similarity(
            grad_smooth[motion_idx].view(-1),
            grad_raw[motion_idx].view(-1),
            dim=0,
        ).item()
        metrics['gradient_correlation_moving'] = grad_corr

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<50} {value:>10.6f}")

    print("=" * 60)
    print("\nInterpretation:")

    if 'temporal_variance_reduction_static' in metrics:
        r   = metrics['temporal_variance_reduction_static']
        tag = "✓ Excellent" if r > 0.5 else ("✓ Good" if r > 0.3 else "⚠ Insufficient")
        print(f"{tag} temporal smoothing: {100*r:.1f}% variance reduction on static pixels")

    if 'flicker_reduction' in metrics:
        r    = metrics['flicker_reduction']
        frac = metrics.get('flicker_pixel_fraction', 0)
        tag  = "✓ Excellent" if r > 0.5 else ("✓ Good" if r > 0.3 else "⚠ Insufficient")
        print(f"{tag} flicker reduction: {100*r:.1f}% "
              f"({100*frac:.1f}% of pixel-pairs flagged as flicker)")

    if 'spatial_gradient_error' in metrics:
        e   = metrics['spatial_gradient_error']
        tag = "✓" if e < 0.01 else "⚠"
        print(f"{tag} Spatial geometry preserved — gradient error: {e:.6f}")

    if 'mae_static' in metrics:
        mae = metrics['mae_static']
        print(f"{'✓' if mae <= 0.05 else '⚠'} MAE on static pixels: {mae:.6f}")

    if 'gradient_correlation_moving' in metrics:
        c   = metrics['gradient_correlation_moving']
        tag = "✓ Excellent" if c > 0.95 else ("✓ Good" if c > 0.85 else "⚠ Could be improved")
        print(f"{tag} motion preservation: {c:.4f} gradient correlation on moving pixels")


def visualize_results(depth_raw: np.ndarray,
                      depth_smooth: np.ndarray,
                      rgb: np.ndarray,
                      output_path: str = 'evaluation.png') -> None:
    """Visualize before/after smoothing."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    t_frames  = [0, depth_raw.shape[0] // 2, depth_raw.shape[0] - 1]

    for col, t in enumerate(t_frames):
        im0 = axes[0, col].imshow(depth_raw[t], cmap='viridis')
        axes[0, col].set_title(f'Raw (frame {t})')
        plt.colorbar(im0, ax=axes[0, col])

        im1 = axes[1, col].imshow(depth_smooth[t], cmap='viridis')
        axes[1, col].set_title(f'Smoothed (frame {t})')
        plt.colorbar(im1, ax=axes[1, col])

        diff = np.abs(depth_smooth[t] - depth_raw[t])
        im2  = axes[2, col].imshow(diff, cmap='hot')
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
    parser = argparse.ArgumentParser(description='Evaluate temporal depth smoothing')
    parser.add_argument('--checkpoint',       required=True)
    parser.add_argument('--depth-input',      required=True)
    parser.add_argument('--rgb-input',        required=True)
    parser.add_argument('--vda-input',        default=None,
                        help='Aligned VDA depth — enables flicker metrics')
    parser.add_argument('--output-dir',       default='./eval_results')
    parser.add_argument('--motion-threshold', type=float, default=0.05)
    parser.add_argument('--depth-threshold',  type=float, default=0.01)
    parser.add_argument('--vda-weight',       type=float, default=0.7)
    parser.add_argument('--save-smoothed',    action='store_true')
    parser.add_argument('--visualize',        action='store_true')
    parser.add_argument('--device',           default='cuda')
    parser.add_argument('--window-size',      type=int, default=16,
                        help='Chunk size for inference (frames). Use smaller values for limited VRAM.')
    parser.add_argument('--overlap',          type=int, default=4,
                        help='Edge overlap per chunk. Should match inference settings.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    depth_raw = np.load(args.depth_input)
    rgb       = np.load(args.rgb_input)
    depth_vda = np.load(args.vda_input) if args.vda_input else None
    print(f"  Depth: {depth_raw.shape}  RGB: {rgb.shape}")
    if depth_vda is not None:
        print(f"  VDA:   {depth_vda.shape}")

    if rgb.max() > 1:
        rgb = rgb / 255.0

    print("\nRunning inference...")
    smoother     = DepthSmoother(args.checkpoint, device=args.device)
    depth_smooth = smoother.smooth(depth_raw, rgb,
                                   window_size=args.window_size,
                                   overlap=args.overlap)

    print("\nEvaluating...")
    metrics = evaluate_smoothing(
        depth_raw, depth_smooth, rgb,
        depth_vda=depth_vda,
        motion_threshold=args.motion_threshold,
        depth_threshold=args.depth_threshold,
        vda_weight=args.vda_weight,
    )
    print_metrics(metrics)

    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()}, f, indent=2)

    if args.save_smoothed:
        out = output_dir / 'depth_smooth.npy'
        np.save(out, depth_smooth)
        print(f"\nSaved smoothed depth to {out}")

    if args.visualize:
        visualize_results(depth_raw, depth_smooth, rgb,
                          str(output_dir / 'visualization.png'))


if __name__ == '__main__':
    main()