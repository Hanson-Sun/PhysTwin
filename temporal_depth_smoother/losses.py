"""Loss functions for temporal depth smoothing."""

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3) / 8.0


def compute_image_gradient(rgb: torch.Tensor) -> torch.Tensor:
    """
    Sobel gradient magnitude of RGB frames.
    Used to build edge-aware weights: low gradient = flat region = trust DA3 depth.

    Args:
        rgb: [B, T, H, W, 3]  float, range [0, 1]
    Returns:
        [B, T, H, W] gradient magnitude
    """
    B, T, H, W, C = rgb.shape
    gray = rgb.view(B * T, C, H, W).mean(dim=1, keepdim=True)

    sx = _SOBEL_X.to(gray.device)
    sy = _SOBEL_Y.to(gray.device)
    grad = (F.conv2d(gray, sx, padding=1) ** 2 +
            F.conv2d(gray, sy, padding=1) ** 2).sqrt().squeeze(1)

    return grad.view(B, T, H, W)


def compute_motion_mask(rgb: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Binary motion mask from frame differencing.
    1 = moving pixel, 0 = static pixel.
    NOTE: expects rgb in [0, 1] — threshold is meaningless otherwise.

    Args:
        rgb: [B, T, H, W, 3]  float, range [0, 1]
    Returns:
        [B, T-1, H, W]
    """
    diff = rgb[:, 1:].float() - rgb[:, :-1].float()
    return (diff.abs().mean(dim=-1) > threshold).float()


def compute_background_mask(depth_raw: torch.Tensor,
                            threshold: float = 0.01) -> torch.Tensor:
    """
    Static background mask from temporal depth variance.
    Used in preprocessing to align VDA scale to DA3 — not used during training.
    Pixels with low depth std over time are assumed to be static background.

    Args:
        depth_raw: [B, T, H, W]
    Returns:
        [B, H, W]  1=background, 0=foreground
    """
    return (depth_raw.std(dim=1) < threshold).float()


def l_temporal(depth_smooth: torch.Tensor,
               depth_vda_aligned: torch.Tensor,
               motion_mask: torch.Tensor,
               motion_weight: float = 0.5) -> torch.Tensor:
    """
    Main training signal. Matches frame-to-frame depth changes to VDA.
    Supervising *gradients* (not absolute values) makes this robust to
    scale misalignment between VDA and DA3.

    Static pixels are fully supervised. Moving pixels are downweighted
    by motion_weight (not zeroed) since VDA on moving objects is noisy
    but not completely useless.

    Args:
        depth_smooth:      [B, T, H, W]
        depth_vda_aligned: [B, T, H, W]
        motion_mask:       [B, T-1, H, W]  1=moving, 0=static
        motion_weight:     supervision weight on moving pixels (0=ignore, 1=full)
    """
    grad_smooth = depth_smooth[:, 1:]      - depth_smooth[:, :-1]
    grad_vda    = depth_vda_aligned[:, 1:] - depth_vda_aligned[:, :-1]
    # Static=1.0, moving=motion_weight
    pixel_weight = 1.0 - (1.0 - motion_weight) * motion_mask
    return (grad_smooth - grad_vda).abs().mul(pixel_weight).mean()


def l_geometric(depth_smooth: torch.Tensor,
                depth_raw: torch.Tensor,
                rgb: torch.Tensor,
                weight_strength: float = 50.0) -> torch.Tensor:
    """
    Keeps output close to raw DA3 depth — prevents over-smoothing.
    Edge-aware: penalty strongest on flat regions, relaxed at RGB edges.

    weight_strength tuning: Sobel magnitudes are typically 0.0–0.1,
    so weight_strength ~50 gives exp(-50*0.05)≈0.08 at edges (relaxed)
    vs exp(0)=1.0 on flat regions (full penalty). Default was 5.0 which
    gave almost no edge relaxation.

    Args:
        depth_smooth: [B, T, H, W]
        depth_raw:    [B, T, H, W]
        rgb:          [B, T, H, W, 3]  float, range [0, 1]
    """
    w = torch.exp(-weight_strength * compute_image_gradient(rgb))
    return (depth_smooth - depth_raw).abs().mul(w).mean()


def l_smooth(depth_smooth: torch.Tensor,
             motion_mask: torch.Tensor) -> torch.Tensor:
    """
    Directly penalizes flicker on static pixels.
    Most targeted loss for the flicker suppression goal — explicitly
    zeroes out any depth change on pixels where RGB shows no motion.

    Args:
        depth_smooth: [B, T, H, W]
        motion_mask:  [B, T-1, H, W]  1=moving, 0=static
    """
    grad = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return grad.mul(1 - motion_mask).mean()


def total_loss(depth_smooth: torch.Tensor,
               depth_raw: torch.Tensor,
               depth_vda_aligned: torch.Tensor,
               rgb: torch.Tensor,
               lambda_temporal: float = 1.0,
               lambda_geometric: float = 0.5,
               lambda_smooth: float = 0.3,
               motion_threshold: float = 0.05,
               motion_weight: float = 0.5) -> dict:
    """
    Args:
        depth_smooth:      [B, T, H, W]  model output
        depth_raw:         [B, T, H, W]  DA3 input
        depth_vda_aligned: [B, T, H, W]  VDA temporal supervision
        rgb:               [B, T, H, W, 3]  float [0, 1]
    Returns:
        dict with 'total', 'temporal', 'geometric', 'smooth'
    """
    # Normalise RGB defensively — motion_mask and Sobel depend on [0,1] range
    rgb_01 = rgb.float().div(255.0) if rgb.max() > 1.0 else rgb.float()

    motion_mask = compute_motion_mask(rgb_01, threshold=motion_threshold)

    loss_t = l_temporal( depth_smooth, depth_vda_aligned, motion_mask, motion_weight)
    loss_g = l_geometric(depth_smooth, depth_raw, rgb_01)
    loss_s = l_smooth(   depth_smooth, motion_mask)

    total = lambda_temporal * loss_t + lambda_geometric * loss_g + lambda_smooth * loss_s

    return {
        'total':     total,
        'temporal':  loss_t.detach(),
        'geometric': loss_g.detach(),
        'smooth':    loss_s.detach(),
    }