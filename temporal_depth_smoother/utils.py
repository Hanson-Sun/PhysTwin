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
    std = depth.std(dim=tuple(range(1, depth.ndim)), keepdim=True)
    std = std.clamp(min=std_clamp)
    
    normalized = (depth - mean) / std
    
    return normalized, mean, std


def denormalize_depth(depth_norm: torch.Tensor, 
                      mean: torch.Tensor, 
                      std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize depth maps using stored mean and std.
    
    Args:
        depth_norm: normalized depth
        mean: per-clip mean
        std: per-clip std
    
    Returns:
        denormalized depth in original scale
    """
    return depth_norm * std + mean


def align_vda_to_da3(depth_vda: torch.Tensor,
                     depth_da3: torch.Tensor,
                     bg_mask: torch.Tensor) -> torch.Tensor:
    """
    Align VDA depth to DA3 using scale/shift on static background pixels.
    
    Solves for s, b in: depth_vda_aligned = s * depth_vda + b
    using least squares on background-only pixels (where static assumption holds).
    
    Args:
        depth_vda: [T, H, W] VDA depth (wrong scale, temporally smooth)
        depth_da3: [T, H, W] DA3 depth (correct geometry, temporally jittery)
        bg_mask: [H, W] background mask (1=static, 0=foreground)
    
    Returns:
        depth_vda_aligned: [T, H, W] VDA aligned to DA3's scale
    """
    T, H, W = depth_vda.shape
    device = depth_vda.device
    
    # Flatten background pixels
    bg_idx = (bg_mask > 0.5).nonzero(as_tuple=True)
    
    if len(bg_idx[0]) < 10:
        # Insufficient background: return raw VDA
        return depth_vda
    
    # Extract background regions
    vda_bg = depth_vda[:, bg_idx[0], bg_idx[1]].view(T, -1)  # [T, N_bg]
    da3_bg = depth_da3[:, bg_idx[0], bg_idx[1]].view(T, -1)
    
    # Least squares: minimize ||s * vda_bg + b - da3_bg||^2
    # Stack all pixels over time
    vda_all = vda_bg.view(-1, 1).float()  # [T*N_bg, 1]
    da3_all = da3_bg.view(-1).float()      # [T*N_bg]
    
    # Add bias term
    ones = torch.ones_like(vda_all)
    X = torch.cat([vda_all, ones], dim=1)  # [T*N_bg, 2]
    
    # Solve: X @ [s, b] = da3_all
    try:
        params = torch.linalg.lstsq(X, da3_all.unsqueeze(1)).solution.squeeze()
        s, b = params[0].item(), params[1].item()
    except:
        # Fallback: identity scale
        s, b = 1.0, 0.0
    
    depth_vda_aligned = s * depth_vda + b
    
    return depth_vda_aligned


def create_raised_cosine_window(T: int, 
                                device: torch.device = None) -> torch.Tensor:
    """
    Create raised cosine window for smooth blending in sliding window inference.
    
    Raised cosine has soft edges — frames near chunk boundaries contribute
    less to the final output, reducing boundary artifacts.
    
    Args:
        T: window length
        device: torch device
    
    Returns:
        window: [T] raised cosine window
    """
    if device is None:
        device = torch.device('cpu')
    
    t = torch.arange(T, dtype=torch.float32, device=device)
    window = 0.5 * (1 - torch.cos(2 * np.pi * t / (T - 1)))
    
    return window


def sliding_window_inference(model: torch.nn.Module,
                             depth_raw: torch.Tensor,
                             rgb: torch.Tensor,
                             window_size: int = 16,
                             stride: int = 8,
                             blend_window: bool = True) -> torch.Tensor:
    """
    Process long videos via sliding window with overlap blending.
    
    Args:
        model: trained TemporalDepthSmoother
        depth_raw: [N, H, W] or [N] full video depth
        rgb: [N, H, W, 3] full video RGB
        window_size: temporal window for model (T)
        stride: frames between window starts
        blend_window: use raised cosine blending at boundaries
    
    Returns:
        output: [N, H, W] smoothed depth for full video
    """
    device = depth_raw.device
    N = depth_raw.shape[0]
    output = torch.zeros_like(depth_raw)
    weights = torch.zeros(N, device=device)
    
    if blend_window:
        window = create_raised_cosine_window(window_size, device=device)
    else:
        window = torch.ones(window_size, device=device)
    
    with torch.no_grad():
        for start in range(0, N - window_size + 1, stride):
            end = start + window_size
            
            chunk_depth = depth_raw[start:end].unsqueeze(0)   # [1, T, H, W]
            chunk_rgb = rgb[start:end].unsqueeze(0)            # [1, T, H, W, 3]
            
            smoothed = model(chunk_depth, chunk_rgb).squeeze(0)  # [T, H, W]
            
            output[start:end] += smoothed * window.view(-1, 1, 1)
            weights[start:end] += window
    
    # Normalize by accumulated weights to avoid over-brightening at overlaps
    output = output / weights.clamp(min=1e-6).view(-1, 1, 1)
    
    return output


def compute_temporal_variance(depth: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal variance of depth per pixel.
    
    Useful for evaluating smoothing: temporal variance on static regions
    should decrease after applying the network.
    
    Args:
        depth: [T, H, W] or [B, T, H, W] depth maps
    
    Returns:
        variance: [H, W] or [B, H, W] per-pixel temporal variance
    """
    if depth.ndim == 4:
        var = depth.var(dim=1)  # [B, H, W]
    else:
        var = depth.var(dim=0)  # [H, W]
    
    return var


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    filepath: str) -> None:
    """Save training checkpoint including model config for inference."""
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss':            loss,
        'config':          model.config,
    }, filepath)


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    filepath: str) -> int:
    """
    Load training checkpoint.
    Returns starting epoch for resuming training.
    """
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'] + 1 