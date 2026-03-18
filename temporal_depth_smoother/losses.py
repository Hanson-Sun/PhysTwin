"""Loss functions for temporal depth smoothing."""

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
 
 
def compute_image_gradient(rgb: torch.Tensor) -> torch.Tensor:
    """
    Sobel gradient magnitude of RGB frames.
 
    Args:
        rgb: [B, T, H, W, 3]  float, range [0, 1]
    Returns:
        [B, T, H, W] gradient magnitude
    """
    B, T, H, W, C = rgb.shape
    gray = rgb.view(B * T, C, H, W).mean(dim=1, keepdim=True)
    sx   = _SOBEL_X.to(gray.device)
    sy   = _SOBEL_Y.to(gray.device)
    grad = (F.conv2d(gray, sx, padding=1) ** 2 +
            F.conv2d(gray, sy, padding=1) ** 2).sqrt().squeeze(1)
    return grad.view(B, T, H, W)
 
 
def compute_depth_spatial_gradient(depth: torch.Tensor) -> tuple:
    """
    Spatial gradients of a depth map using finite differences.
 
    Args:
        depth: [B, T, H, W]
    Returns:
        grad_x: [B, T, H, W]  horizontal gradient
        grad_y: [B, T, H, W]  vertical gradient
    """
    grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]  # [B, T, H, W-1]
    grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]  # [B, T, H-1, W]
    grad_x = F.pad(grad_x, (0, 1))        # pad right  → [B, T, H, W]
    grad_y = F.pad(grad_y, (0, 0, 0, 1))  # pad bottom → [B, T, H, W]
    return grad_x, grad_y


def compute_motion_mask(rgb: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Binary motion mask from RGB frame differencing.
    1 = moving pixel, 0 = static pixel.
    NOTE: expects rgb in [0, 1].

    Args:
        rgb: [B, T, H, W, 3]  float, range [0, 1]
    Returns:
        [B, T-1, H, W]
    """
    diff = rgb[:, 1:].float() - rgb[:, :-1].float()
    return (diff.abs().mean(dim=-1) > threshold).float()


def compute_flicker_mask(depth_raw: torch.Tensor,
                         depth_vda: torch.Tensor,
                         rgb: torch.Tensor,
                         depth_threshold: float = 0.01,
                         rgb_threshold: float = 0.05,
                         vda_weight: float = 0.7) -> torch.Tensor:
    """
    Robust flicker mask combining VDA and RGB signals.
 
    A pixel is flickering if DA3 changes between frames but VDA and/or
    RGB say nothing is moving there.
 
    Args:
        depth_raw:       [B, T, H, W]   DA3 raw depth
        depth_vda:       [B, T, H, W]   VDA aligned depth
        rgb:             [B, T, H, W, 3] float [0, 1]
        depth_threshold: threshold for significant depth change (normalised)
        rgb_threshold:   threshold for significant RGB change
        vda_weight:      how much to trust VDA vs RGB (0=RGB only, 1=VDA only)
    Returns:
        flicker_mask: [B, T-1, H, W]  soft values in [0, 1]
    """
    depth_std       = depth_raw.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
    da3_change      = (depth_raw[:, 1:] - depth_raw[:, :-1]).abs()
    da3_change_norm = da3_change / depth_std
 
    vda_std         = depth_vda.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
    vda_change      = (depth_vda[:, 1:] - depth_vda[:, :-1]).abs()
    vda_change_norm = vda_change / vda_std
    vda_static      = (vda_change_norm < depth_threshold).float()
 
    rgb_diff   = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(dim=-1)
    rgb_static = (rgb_diff < rgb_threshold).float()
 
    combined_static = vda_weight * vda_static + (1.0 - vda_weight) * rgb_static
    has_da3_change  = (da3_change_norm > depth_threshold * 0.5).float()
 
    return combined_static * has_da3_change  # [B, T-1, H, W]

def l_flicker(depth_smooth: torch.Tensor,
              flicker_mask: torch.Tensor) -> torch.Tensor:
    """
    Suppress depth changes on pixels identified as flickering.

    Args:
        depth_smooth:  [B, T, H, W]
        flicker_mask:  [B, T-1, H, W]  soft mask, 1=flicker, 0=real motion
    Returns:
        scalar loss
    """
    change = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return change.mul(flicker_mask).mean()


def l_geometric(depth_smooth: torch.Tensor,
                depth_raw: torch.Tensor,
                rgb: torch.Tensor,
                edge_weight_strength: float = 10.0) -> torch.Tensor:
    """
    Preserve spatial depth structure of DA3, relaxed at RGB edges.

    Flat regions: spatial gradients are reliable → enforce strongly
    Edge regions: spatial gradients are noisy/uncertain → relax

    Args:
        depth_smooth:        [B, T, H, W]
        depth_raw:           [B, T, H, W]
        rgb:                 [B, T, H, W, 3]  float [0, 1]
        edge_weight_strength: controls relaxation at edges.
                              higher = more relaxation at edges.
    """
    smooth_gx, smooth_gy = compute_depth_spatial_gradient(depth_smooth)
    raw_gx,    raw_gy    = compute_depth_spatial_gradient(depth_raw)

    # Edge weight: 1.0 on flat regions, near 0 at RGB edges
    edge_w = torch.exp(-edge_weight_strength * compute_image_gradient(rgb))

    loss_x = (smooth_gx - raw_gx).abs().mul(edge_w).mean()
    loss_y = (smooth_gy - raw_gy).abs().mul(edge_w).mean()

    return (loss_x + loss_y) / 2.0


def l_smooth(depth_smooth: torch.Tensor,
             motion_mask: torch.Tensor) -> torch.Tensor:
    """
    Kept as a lightweight auxiliary loss on RGB-static pixels.
    Less central than before — l_flicker is now the main flicker signal.

    Args:
        depth_smooth: [B, T, H, W]
        motion_mask:  [B, T-1, H, W]  1=moving, 0=static (from RGB)
    """
    grad = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return grad.mul(1 - motion_mask).mean()


def total_loss(depth_smooth: torch.Tensor,
               depth_raw: torch.Tensor,
               depth_vda_aligned: torch.Tensor,
               rgb: torch.Tensor,
               lambda_flicker: float = 1.0,
               lambda_geometric: float = 1.0,
               lambda_smooth: float = 0.01,
               depth_threshold: float = 0.01,
               rgb_threshold: float = 0.05,
               vda_weight: float = 0.7) -> dict:
    """
    Args:
        depth_smooth:      [B, T, H, W]  model output
        depth_raw:         [B, T, H, W]  DA3 input
        depth_vda_aligned: [B, T, H, W]  VDA (flicker detection only)
        rgb:               [B, T, H, W, 3]  float [0, 1]
        lambda_flicker:    weight for flicker suppression
        lambda_geometric:  weight for spatial geometry preservation
        lambda_smooth:     auxiliary RGB smoothness (0 = disabled)
    Returns:
        dict with 'total', 'flicker', 'geometric', 'smooth'
    """
    rgb_01 = rgb.float().div(255.0) if rgb.max() > 1.0 else rgb.float()
 
    flicker_mask = compute_flicker_mask(
        depth_raw, depth_vda_aligned, rgb_01,
        depth_threshold=depth_threshold,
        rgb_threshold=rgb_threshold,
        vda_weight=vda_weight,
    )
 
    loss_f = l_flicker(depth_smooth, flicker_mask)
    loss_g = l_geometric(depth_smooth, depth_raw, rgb_01)
    loss_s = torch.tensor(0.0, device=depth_smooth.device)
    if lambda_smooth > 0:
        motion_mask = compute_motion_mask(rgb_01, threshold=rgb_threshold)
        grad        = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
        loss_s      = grad.mul(1 - motion_mask).mean()
 
    total = lambda_flicker * loss_f + lambda_geometric * loss_g + lambda_smooth * loss_s
 
    return {
        'total':     total,
        'flicker':   loss_f.detach(),
        'geometric': loss_g.detach(),
        'smooth':    loss_s.detach(),
    }