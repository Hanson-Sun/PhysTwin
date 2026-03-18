"""Utility functions for temporal depth smoothing."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def normalize_depth_clip(depth: torch.Tensor,
                         std_clamp: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-clip normalization of depth maps.

    Args:
        depth: [B, T, H, W] or [T, H, W] depth maps
        std_clamp: minimum std to prevent division by zero

    Returns:
        normalized: normalized depth
        mean: per-clip mean
        std: per-clip std
    """
    mean = depth.mean(dim=tuple(range(1, depth.ndim)), keepdim=True)
    std  = depth.std( dim=tuple(range(1, depth.ndim)), keepdim=True).clamp(min=std_clamp)
    return (depth - mean) / std, mean, std


def denormalize_depth(depth_norm: torch.Tensor,
                      mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:
    """Denormalize depth maps using stored mean and std."""
    return depth_norm * std + mean


def align_vda_to_da3(depth_vda: torch.Tensor,
                     depth_da3: torch.Tensor,
                     bg_mask: torch.Tensor = None,
                     low_pct: float = 2.0,
                     high_pct: float = 98.0) -> torch.Tensor:
    """
    Align VDA depth to DA3 using quantile normalization.

    Both VDA and DA3 are mapped to the same quantile range [0, 1] using
    their own per-clip percentiles. This is robust to:
      - Wrong scale (VDA outputs 0-1 while DA3 is in metres)
      - Wrong shift (affine offset)
      - Flipped depth / disparity
      - Completely broken VDA values (near-zero, huge range, etc.)

    Unlike least-squares on background pixels, this method:
      - Does not depend on background mask quality
      - Does not fail when VDA range is near-zero
      - Does not require static background pixels
      - Always produces output in the same [0, 1] range as DA3

    After alignment both depth maps are in [~0, ~1] space.
    The model's per-clip normalization in model.py handles the rest.

    Args:
        depth_vda:  [T, H, W] VDA depth (any scale)
        depth_da3:  [T, H, W] DA3 depth (any scale)
        bg_mask:    ignored — kept for backward compatibility with preprocess.py
        low_pct:    lower percentile for robust min (default 2)
        high_pct:   upper percentile for robust max (default 98)

    Returns:
        depth_vda_aligned: [T, H, W] VDA mapped to DA3's quantile space
        depth_da3_normed:  [T, H, W] DA3 mapped to [0, 1]  (save this as depth_raw)
    """
    def quantile_normalize(depth: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Map depth to [0,1] using robust percentiles. Returns (normed, lo, hi).
        Uses numpy percentile to avoid torch.quantile's 16M element limit."""
        arr  = depth.reshape(-1).float().numpy()
        lo   = float(np.percentile(arr, low_pct))
        hi   = float(np.percentile(arr, high_pct))
        span = hi - lo
        if span < 1e-6:
            return torch.zeros_like(depth), lo, hi
        normed = (depth.float() - lo) / span
        return normed, lo, hi

    vda_normed, vda_lo, vda_hi = quantile_normalize(depth_vda)
    da3_normed, da3_lo, da3_hi = quantile_normalize(depth_da3)

    return vda_normed, da3_normed


def align_vda_to_da3_legacy(depth_vda: torch.Tensor,
                             depth_da3: torch.Tensor,
                             bg_mask: torch.Tensor) -> torch.Tensor:
    """
    Original least-squares alignment on background pixels.
    Kept for reference — use align_vda_to_da3 instead.
    """
    T, H, W = depth_vda.shape
    bg_idx  = (bg_mask > 0.5).nonzero(as_tuple=True)

    if len(bg_idx[0]) < 10:
        return depth_vda

    vda_bg  = depth_vda[:, bg_idx[0], bg_idx[1]].view(T, -1)
    da3_bg  = depth_da3[:, bg_idx[0], bg_idx[1]].view(T, -1)
    vda_all = vda_bg.view(-1, 1).float()
    da3_all = da3_bg.view(-1).float()
    ones    = torch.ones_like(vda_all)
    X       = torch.cat([vda_all, ones], dim=1)

    try:
        params = torch.linalg.lstsq(X, da3_all.unsqueeze(1)).solution.squeeze()
        s, b   = params[0].item(), params[1].item()
    except Exception:
        s, b = 1.0, 0.0

    return s * depth_vda + b


def create_raised_cosine_window(T: int,
                                device: torch.device = None) -> torch.Tensor:
    """
    Create raised cosine window for smooth blending in sliding window inference.

    Args:
        T: window length
        device: torch device

    Returns:
        window: [T] raised cosine window
    """
    if device is None:
        device = torch.device('cpu')
    t = torch.arange(T, dtype=torch.float32, device=device)
    return 0.5 * (1 - torch.cos(2 * np.pi * t / (T - 1)))


def sliding_window_inference(model: torch.nn.Module,
                             depth_raw: torch.Tensor,
                             rgb: torch.Tensor,
                             window_size: int = 16,
                             stride: int = 8,
                             blend_window: bool = True) -> torch.Tensor:
    """
    Process long videos via sliding window with overlap blending.

    Args:
        model:       trained TemporalDepthSmoother
        depth_raw:   [N, H, W] full video depth
        rgb:         [N, H, W, 3] full video RGB
        window_size: temporal window for model (T)
        stride:      frames between window starts
        blend_window: use raised cosine blending at boundaries

    Returns:
        output: [N, H, W] smoothed depth for full video
    """
    device  = depth_raw.device
    N       = depth_raw.shape[0]
    output  = torch.zeros_like(depth_raw)
    weights = torch.zeros(N, device=device)

    window = create_raised_cosine_window(window_size, device=device) if blend_window \
             else torch.ones(window_size, device=device)

    with torch.no_grad():
        for start in range(0, N - window_size + 1, stride):
            end         = start + window_size
            chunk_depth = depth_raw[start:end].unsqueeze(0)
            chunk_rgb   = rgb[start:end].unsqueeze(0)
            smoothed    = model(chunk_depth, chunk_rgb).squeeze(0)
            output[start:end]  += smoothed * window.view(-1, 1, 1)
            weights[start:end] += window

    return output / weights.clamp(min=1e-6).view(-1, 1, 1)


def compute_temporal_variance(depth: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal variance of depth per pixel.

    Args:
        depth: [T, H, W] or [B, T, H, W]

    Returns:
        variance: [H, W] or [B, H, W]
    """
    return depth.var(dim=1) if depth.ndim == 4 else depth.var(dim=0)


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    filepath: str) -> None:
    """Save training checkpoint including model config for inference."""
    actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss':            loss,
        'config':          actual_model.config,
    }, filepath)


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    filepath: str) -> int:
    """
    Load training checkpoint.
    Returns starting epoch for resuming training.
    Works with both compiled and non-compiled models.
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'] + 1