"""Loss functions for temporal depth smoothing."""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from kornia.filters import gaussian_blur2d



_SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_sobel_cache: dict = {}


def _get_sobel(device):
    if device not in _sobel_cache:
        _sobel_cache[device] = (_SOBEL_X.to(device), _SOBEL_Y.to(device))
    return _sobel_cache[device]


def compute_image_gradient(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Sobel gradient magnitude of RGB frames.

    Args:
        rgb: [B, T, H, W, 3] float, range [0, 1]
    Returns:
        [B, T, H, W] gradient magnitude
    """
    B, T, H, W, C = rgb.shape

    # [B, T, 3, H, W] -> [B*T, 3, H, W]
    rgb_btchw = rgb.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float()

    # convert to grayscale
    gray = rgb_btchw.mean(dim=1, keepdim=True)  # [B*T, 1, H, W]

    sx, sy = _get_sobel(gray.device)
    grad_x = F.conv2d(gray, sx, padding=1)
    grad_y = F.conv2d(gray, sy, padding=1)
    grad = (grad_x ** 2 + grad_y ** 2 + eps).sqrt()

    return grad.reshape(B, T, H, W)


def compute_depth_gradient_sobel(depth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Sobel gradient magnitude of a depth map.

    Args:
        depth: [B, T, H, W]
    Returns:
        [B, T, H, W] gradient magnitude
    """
    B, T, H, W = depth.shape
    d = depth.view(B * T, 1, H, W)
    sx, sy = _get_sobel(d.device)
    grad = (F.conv2d(d, sx, padding=1) ** 2 + F.conv2d(d, sy, padding=1) ** 2 + eps).sqrt()
    return grad.view(B, T, H, W)


def compute_depth_spatial_gradient(depth: torch.Tensor) -> tuple:
    """
    Spatial gradients of a depth map using finite differences.

    Args:
        depth: [B, T, H, W]
    Returns:
        grad_x: [B, T, H, W]
        grad_y: [B, T, H, W]
    """
    grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    grad_x = F.pad(grad_x, (0, 1))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
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


def compute_flicker_mask(
    depth_raw: torch.Tensor,
    depth_vda: torch.Tensor,
    rgb: torch.Tensor,
    depth_threshold: float = 0.15,
    rgb_threshold: float = 0.05,
    vda_weight: float = 0.7,
) -> torch.Tensor:
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
    depth_std = depth_raw.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
    da3_change = (depth_raw[:, 1:] - depth_raw[:, :-1]).abs()
    da3_change_norm = da3_change / depth_std

    vda_std = depth_vda.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
    vda_change = (depth_vda[:, 1:] - depth_vda[:, :-1]).abs()
    vda_change_norm = vda_change / vda_std
    vda_static = (vda_change_norm < depth_threshold).float()

    rgb_diff = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(dim=-1)
    rgb_static = (rgb_diff < rgb_threshold).float()

    combined_static = vda_weight * vda_static + (1.0 - vda_weight) * rgb_static
    has_da3_change = (da3_change_norm > depth_threshold * 0.5).float()

    return combined_static * has_da3_change


# ── Normalisation baselines ────────────────────────────────────────────────────
#
# Each loss is divided by a baseline that puts it on a comparable scale.
# The goal is for all losses to start near 1.0 at initialisation, so that
# lambda values have consistent meaning regardless of the loss's natural magnitude.
#
# Baseline strategy per loss type:
#   TGM:      identity baseline is non-zero (DA3/VDA disagreement) → use directly.
#   TV:       identity baseline is DA3's own second derivative (not VDA's — VDA is
#             too smooth, gives near-zero and explodes the normalised value).
#   Fidelity: identity residual = 0 (output=raw → exact match), so normalise by
#             the aligned depth's std — a residual of 1.0 = one std off.
#   Geometric: identity residual = 0 (same reason), so normalise by DA3's own
#             spatial gradient magnitude — distortion measured as fraction of
#             DA3's natural spatial structure.
#
# All baselines are fully detached — they are scaling constants, not learned.
                
def _tgm_baseline(
    depth_raw: torch.Tensor, depth_vda: torch.Tensor, k: int = 1
) -> torch.Tensor:
    """
    TGM at identity = |grad_raw - grad_vda| — the irreducible DA3/VDA
    temporal disagreement. Non-zero and meaningful: at step 0 with an
    untrained model, normalised TGM starts at exactly 1.0. Reducing
    below 1.0 means the model is improving on the identity baseline.
    """
    with torch.no_grad():
        grad_raw = depth_raw[:, k:] - depth_raw[:, :-k]
        grad_vda = depth_vda[:, k:] - depth_vda[:, :-k]
        return torch.abs(grad_raw - grad_vda).mean().detach().clamp(min=1e-6)


def _tv_baseline(
    depth_raw: torch.Tensor, depth_vda: torch.Tensor, k: int = 1
) -> torch.Tensor:
    """
    TV at identity = DA3's own second derivative, gated by VDA's acceleration.
    Uses DA3 (not VDA) because VDA is very smooth — its second derivative is
    near zero, which would inflate the normalised TV loss to very large values.
    """
    with torch.no_grad():
        d2_raw = depth_raw[:, 2 * k :] - 2 * depth_raw[:, k:-k] + depth_raw[:, : -2 * k]
        d2_vda = (
            depth_vda[:, 2 * k :] - 2 * depth_vda[:, k:-k] + depth_vda[:, : -2 * k]
        ).abs()
        weight = torch.exp(-10.0 * d2_vda)
        return d2_raw.abs().mul(weight).mean().detach().clamp(min=1e-6)


def _fidelity_baseline(
    depth_raw: torch.Tensor, t: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Fidelity identity residual = 0, so we use the std of the aligned raw
    depth as the scale reference. A fidelity loss of 1.0 means the output
    deviates from DA3 by one standard deviation in aligned space.
    """
    with torch.no_grad():
        raw_aligned = (depth_raw - t.detach()) / s.detach()
        return raw_aligned.std().detach().clamp(min=1e-6)


def l_fidelity(
    depth_smooth: torch.Tensor,
    depth_raw: torch.Tensor,
    depth_vda: torch.Tensor,
    gate_floor: float = 0.25,
) -> torch.Tensor:
    """
    Median-aligned fidelity loss, gated by VDA temporal stability.
    Normalised so a fully-wrong output produces loss ~1.0.

    Clip-level alignment via DA3's temporal median removes clip-to-clip
    scale variation while keeping per-frame absolute values intact, so
    fidelity and TGM operate in the same space.

    Gated by VDA's temporal gradient: stable regions (small grad_vda)
    get full enforcement; dynamic regions are relaxed to gate_floor.
    VDA is used only as a motion oracle — not its depth values — so
    VDA's wrong geometry does not contaminate the signal.

    Args:
        depth_smooth: [B, T, H, W]  model output
        depth_raw:    [B, T, H, W]  DA3 input (correct geometry, flickery)
        depth_vda:    [B, T, H, W]  VDA input (motion oracle only)
        gate_floor:   minimum gate value — prevents full relaxation
    Returns:
        scalar loss, normalised by aligned depth std
    """
    B, T, H, W = depth_smooth.shape

    # ── Clip-level alignment (shared coordinate system) ──
    raw_median = depth_raw.median(dim=1, keepdim=True).values  # [B, 1, H, W]
    flat = raw_median.reshape(B, -1)
    t = flat.median(dim=-1).values.view(B, 1, 1, 1)
    s = (flat - t.reshape(B, 1)).abs().mean(dim=-1).view(B, 1, 1, 1).clamp(min=1e-6)

    smooth_aligned = (depth_smooth - t.detach()) / s.detach()
    raw_aligned = (depth_raw - t.detach()) / s.detach()
    residual = torch.abs(smooth_aligned - raw_aligned)  # [B, T, H, W]

    # ── VDA-based temporal gate ──
    grad_vda = (depth_vda[:, 1:] - depth_vda[:, :-1]).abs()  # [B, T-1, H, W]
    grad_vda_norm = grad_vda / (grad_vda.mean() + 1e-6)
    gate_pairs = gate_floor + (1.0 - gate_floor) * torch.exp(-grad_vda_norm)
    gate_frame0 = torch.ones(B, 1, H, W, device=depth_smooth.device)
    gate = torch.cat([gate_frame0, gate_pairs], dim=1)  # [B, T, H, W]

    raw_loss = residual.mul(gate).mean()
    baseline = _fidelity_baseline(depth_raw, t, s)
    return raw_loss / baseline


def l_ssim(
    depth_smooth: torch.Tensor,
    depth_raw: torch.Tensor,
    depth_vda: torch.Tensor,
    gate_floor: float = 0.05,
) -> torch.Tensor:
    B, T, H, W = depth_smooth.shape

    if not torch.isfinite(depth_smooth).all():
        return torch.tensor(0.0, device=depth_smooth.device, requires_grad=False)

    # Clip-level alignment — consistent with l_fidelity
    raw_median = depth_raw.median(dim=1, keepdim=True).values  # [B, 1, H, W]
    flat = raw_median.reshape(B, -1)
    t = flat.median(dim=-1).values.view(B, 1, 1, 1)
    s = (flat - t.reshape(B, 1)).abs().mean(dim=-1).view(B, 1, 1, 1).clamp(min=1e-4)

    smooth_aligned = (depth_smooth - t.detach()) / s.detach()
    target_aligned = (depth_raw - t.detach()) / s.detach()  # [B, T, H, W] per-frame

    # Per-clip data_range
    target_flat = target_aligned.reshape(B, -1)
    per_clip_range = (
        target_flat.max(dim=-1).values - target_flat.min(dim=-1).values
    ).clamp(min=0.1)

    smooth_normed = smooth_aligned / per_clip_range.view(B, 1, 1, 1)
    target_normed = target_aligned / per_clip_range.view(B, 1, 1, 1)

    p = smooth_normed.reshape(B * T, 1, H, W)
    r = target_normed.reshape(B * T, 1, H, W).detach()

    win_size = 7 if min(H, W) >= 7 else 3
    ssim_map = ssim(p, r, data_range=1.0, win_size=win_size, size_average=False)  # [B*T]
    ssim_loss = (1.0 - ssim_map).clamp(min=0.0).reshape(B, T)

    # VDA gate — keep spatial dims, average over H, W AFTER multiplying with ssim_loss
    grad_vda = (depth_vda[:, 1:] - depth_vda[:, :-1]).abs()       # [B, T-1, H, W]
    grad_vda_norm = grad_vda / (grad_vda.mean(dim=[2, 3], keepdim=True) + 1e-6)  # normalise per-frame spatially
    gate_pairs = gate_floor + (1.0 - gate_floor) * torch.exp(-grad_vda_norm)     # [B, T-1, H, W]
    gate_frame0 = torch.ones(B, 1, H, W, device=depth_smooth.device)
    gate = torch.cat([gate_frame0, gate_pairs], dim=1)             # [B, T, H, W]

    # ssim() returns one scalar per frame, so we need a per-frame gate scalar —
    # but now we derive it properly as a weighted mean rather than a blind spatial mean
    gate_per_frame = gate.mean(dim=[2, 3])                         # [B, T] — valid now because gate is spatially meaningful

    loss = (ssim_loss * gate_per_frame).mean()

    if loss.isnan():
        raise ValueError("NaN in ssim_loss * gate_per_frame")

    return loss


def l_flicker(depth_smooth: torch.Tensor, flicker_mask: torch.Tensor) -> torch.Tensor:
    """
    Suppress depth changes on pixels identified as flickering.
    Not normalised — only active when lambda_flicker > 0.
    """
    change = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return change.mul(flicker_mask).mean()


def l_tgm_k(
    depth_smooth: torch.Tensor,
    depth_vda: torch.Tensor,
    depth_raw: torch.Tensor,
    k: int = 1,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    TGM with frame skip k — matches depth change between frame t and t+k.
    Not normalised (baseline depends on k) — use a small lambda_tgm to keep it comparable.

    Supervises longer-term consistency. With k=1 it sees only frame-to-frame flicker;
    with larger k it also sees longer-term oscillations that TGM with k=1 cannot detect.

    Args:
        depth_smooth: [B, T, H, W]  model output
        depth_vda:    [B, T, H, W]  VDA aligned depth
        depth_raw:    [B, T, H, W]  DA3 input (baseline computation only)
        k:            frame skip (e.g., 2 for t vs t+2)
    Returns:
        scalar loss
    """
    grad_smooth = depth_smooth[:, k:] - depth_smooth[:, :-k]
    grad_vda = depth_vda[:, k:] - depth_vda[:, :-k]
    raw_loss = torch.abs(grad_smooth - grad_vda).mean()
    
    if baseline is None:
        baseline = _tgm_baseline(depth_raw, depth_vda, k)
        
    return raw_loss / baseline


def l_tv_k(
    depth_smooth: torch.Tensor,
    depth_vda: torch.Tensor,
    depth_raw: torch.Tensor,
    k: int = 1,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Temporal total variation with frame skip k — matches depth change between frame t and t+2k.
    Not normalised (baseline depends on k) — use a small lambda_tv to keep it comparable.

    Penalises longer-term oscillations that TGM with frame skip k cannot see.

    Args:
        depth_smooth: [B, T, H, W]  model output
        depth_vda:    [B, T, H, W]  VDA aligned depth
        depth_raw:    [B, T, H, W]  DA3 input (baseline computation only)
        k:            frame skip (e.g., 2 for t vs t+4)
    Returns:
        scalar loss
    """
    d2_smooth = depth_smooth[:, 2*k:] - 2 * depth_smooth[:, k:-k] + depth_smooth[:, :-2*k]
    d2_vda    = (depth_vda[:, 2*k:] - 2 * depth_vda[:, k:-k] + depth_vda[:, :-2*k]).abs()
    weight    = torch.exp(-10.0 * d2_vda)
    raw_loss  = d2_smooth.abs().mul(weight).mean()

    if baseline is None:
        baseline = _tv_baseline(depth_raw, depth_vda, k=k)

    return raw_loss / baseline


import matplotlib.pyplot as plt

def l_geometric(
    depth_smooth: torch.Tensor,
    depth_raw: torch.Tensor,
    depth_vda: torch.Tensor,
    rgb: torch.Tensor,
    rgb_edge_threshold: float = 0.6,
    depth_edge_threshold: float = 0.6,
    sharpness_threshold: float = 0.8,
) -> torch.Tensor:
    """
    Edge sharpness loss with robust per-clip normalization.
    
    Normalizes both RGB and depth gradients per-clip to [0, 1] using the 95th 
    percentile, ensuring threshold values work uniformly across different clips 
    with different scales and contrasts.
    
    Args:
        depth_smooth:        [B, T, H, W]  model output
        depth_raw:           [B, T, H, W]  DA3 input (unused)
        depth_vda:           [B, T, H, W]  VDA depth — edge location only
        rgb:                 [B, T, H, W, 3]  float [0, 1]
        rgb_edge_threshold:  normalized RGB gradient threshold for edge detection (0-1)
        depth_edge_threshold: normalized depth gradient threshold for edge detection (0-1)
        sharpness_threshold: normalized smoothed depth gradient floor (0-1)
    Returns:
        scalar loss
    """
    B, T, H, W = depth_smooth.shape
    input_dtype = depth_smooth.dtype

    with torch.no_grad():
        rgb_mag = compute_image_gradient(rgb)  # [B, T, H, W]
        
        # Per-clip 95th percentile normalization: each clip scales to roughly [0, 1]
        rgb_flat = rgb_mag.reshape(B, -1)
        rgb_scale = torch.quantile(rgb_flat, 0.95, dim=-1, keepdim=True)  # [B, 1]
        rgb_scale = rgb_scale.view(B, 1, 1, 1).clamp(min=1e-6)
        rgb_mag_n = rgb_mag / rgb_scale
        rgb_is_edge = (rgb_mag_n > rgb_edge_threshold).float()

        vda_mag = compute_depth_gradient_sobel(depth_vda)
        
        # Per-clip 95th percentile normalization
        vda_flat = vda_mag.reshape(B, -1)
        vda_scale = torch.quantile(vda_flat, 0.95, dim=-1, keepdim=True)  # [B, 1]
        vda_scale = vda_scale.view(B, 1, 1, 1).clamp(min=1e-6)
        vda_mag_n = vda_mag / vda_scale
        vda_is_edge = (vda_mag_n > depth_edge_threshold).float()

        edge_mask = rgb_is_edge * vda_is_edge

    smooth_mag = compute_depth_gradient_sobel(depth_smooth.float())
    smooth_mag_n = smooth_mag / vda_scale.detach()
    
    target_sharpness = sharpness_threshold * vda_mag_n.detach()
    residual = F.relu(target_sharpness - smooth_mag_n) * edge_mask
    raw_loss = residual.sum() / edge_mask.sum().clamp(min=1.0)
    
    # depth_smooth_vis = depth_smooth.detach().cpu()
    # smooth_mag_n_vis = smooth_mag_n.detach().cpu()
    # edge_mask_vis = edge_mask.detach().cpu()
    # residual_vis = residual.detach().cpu()
    # 
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.title("Edge Mask")
    # plt.imshow(edge_mask_vis[0,0].numpy(), cmap='magma')
    # 
    # plt.subplot(1,3,2)
    # plt.title("Sharpness Loss")
    # plt.imshow(residual_vis[0,0].numpy(), cmap='magma')
    # 
    # plt.subplot(1,3,3)
    # plt.title("Masked Edge Magnitude")
    # plt.imshow((smooth_mag_n_vis[0,0]*edge_mask_vis[0,0]).numpy(), cmap='magma')
    # 
    # plt.show()
    
    with torch.no_grad():
        baseline = (vda_mag_n * edge_mask).sum() / edge_mask.sum().clamp(min=1.0)
        baseline = baseline.clamp(min=1e-6)

    return (raw_loss / baseline).to(input_dtype)

def l_geometric_gradient(
    depth_smooth: torch.Tensor,
    depth_raw: torch.Tensor,
    depth_vda: torch.Tensor,
    rgb: torch.Tensor
) -> torch.Tensor:
    smooth_gx, smooth_gy = compute_depth_spatial_gradient(depth_smooth)
    raw_gx,    raw_gy    = compute_depth_spatial_gradient(depth_raw)

    edge_w = compute_depth_gradient_sobel(depth_vda)                           # [B, T, H, W]
    edge_w = edge_w / (edge_w.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
    edge_w = edge_w.clamp(min=0.1)

    loss_x = (smooth_gx - raw_gx).abs().mul(edge_w).mean()
    loss_y = (smooth_gy - raw_gy).abs().mul(edge_w).mean()
    raw_loss = (loss_x + loss_y) / 2.0

    with torch.no_grad():
        baseline_x = raw_gx.abs().mul(edge_w).mean()
        baseline_y = raw_gy.abs().mul(edge_w).mean()
        baseline = ((baseline_x + baseline_y) / 2.0).clamp(min=1e-6)

    return raw_loss / baseline


# deprecated
def l_smooth(depth_smooth: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
    """Deprecated — set lambda_smooth=0."""
    grad = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return grad.mul(1 - motion_mask).mean()


# ── Total loss ─────────────────────────────────────────────────────────────────


def total_loss(
    depth_smooth: torch.Tensor,
    depth_raw: torch.Tensor,
    depth_vda_aligned: torch.Tensor,
    rgb: torch.Tensor,
    lambda_fidelity: float = 1.0,
    lambda_tgm: float = 1.0,
    lambda_geometric: float = 1.0,
    lambda_geometric_grad: float = 1.0,
    lambda_tv: float = 1.0,
    lambda_ssim: float = 1.0,
    depth_threshold: float = 0.15,
    rgb_threshold: float = 0.05,
    vda_weight: float = 0.7,
    tgm_baseline: torch.Tensor | None = None,
    tv_baseline: torch.Tensor | None = None,
) -> dict:
    """
    All losses are normalised to ~1.0 at the identity baseline, so
    lambda=1.0 means this loss contributes one "identity unit" of gradient signal.

    Lambda meaning is consistent across all losses — use similar magnitudes for each.

    Args:
        depth_smooth:      [B, T, H, W]  model output
        depth_raw:         [B, T, H, W]  DA3 input
        depth_vda_aligned: [B, T, H, W]  VDA aligned depth
        rgb:               [B, T, H, W, 3]  uint8 or float [0, 1]
        lambda_fidelity:   geometry preservation weight
        lambda_tgm:        temporal gradient matching weight
        lambda_geometric:  spatial edge sharpness weight
        lambda_geometric_grad:  spatial gradient matching weight
        lambda_tv:         second-order temporal smoothness weight
        lambda_ssim:       SSIM weight
    Returns:
        dict: 'total', 'fidelity', 'tgm', 'geometric', 'geometric_grad', 'tv', 'ssim'
    """
    rgb_01 = rgb.float().div(255.0) if rgb.max() > 1.0 else rgb.float()
    
    loss_fid = torch.tensor(0.0, device=depth_smooth.device)
    loss_g = torch.tensor(0.0, device=depth_smooth.device)
    loss_ggrad = torch.tensor(0.0, device=depth_smooth.device)
    loss_ssim = torch.tensor(0.0, device=depth_smooth.device)
    loss_tv = torch.tensor(0.0, device=depth_smooth.device)
    loss_tgm = torch.tensor(0.0, device=depth_smooth.device)
    
    if lambda_fidelity > 0.0:
        loss_fid = lambda_fidelity * l_fidelity(depth_smooth, depth_raw, depth_vda_aligned)

    if lambda_geometric > 0.0:
        loss_g = lambda_geometric * l_geometric(depth_smooth, depth_raw, depth_vda_aligned, rgb_01)
    
    if lambda_geometric_grad > 0.0:
        loss_ggrad = lambda_geometric_grad * l_geometric_gradient(
            depth_smooth, depth_raw, depth_vda_aligned, rgb_01
        )
        
    if lambda_ssim > 0.0:
        loss_ssim = lambda_ssim * l_ssim(depth_smooth, depth_raw, depth_vda_aligned)
        
    if lambda_tv > 0.0:
        loss_tv_2 = l_tv_k(depth_smooth, depth_vda_aligned, depth_raw, k=2)
        loss_tv_1 = l_tv_k(depth_smooth, depth_vda_aligned, depth_raw, k=1)
        loss_tv = lambda_tv * (loss_tv_1 + 0.25 * loss_tv_2)
        
    if lambda_tgm > 0.0:
        loss_tgm_3 = l_tgm_k(depth_smooth, depth_vda_aligned, depth_raw, k=3)
        loss_tgm_2 = l_tgm_k(depth_smooth, depth_vda_aligned, depth_raw, k=2)
        loss_tgm_1 = l_tgm_k(depth_smooth, depth_vda_aligned, depth_raw, k=1)
        loss_tgm = lambda_tgm * (loss_tgm_1 + 0.25 * loss_tgm_2 + 0.05 * loss_tgm_3)
        
    if loss_fid.isnan().any():
        raise ValueError("loss_fid contains NaN values")
    if loss_g.isnan().any():
        raise ValueError("loss_g contains NaN values")
    if loss_ggrad.isnan().any():
        raise ValueError("loss_ggrad contains NaN values")
    if loss_ssim.isnan().any():
        raise ValueError("loss_ssim contains NaN values")
    if loss_tv.isnan().any():
        raise ValueError("loss_tv contains NaN values")
    if loss_tgm.isnan().any():
        raise ValueError("loss_tgm contains NaN values")
    

    return {
        "total": loss_fid + loss_g + loss_ggrad + loss_ssim + loss_tv + loss_tgm,
        "fidelity": loss_fid.detach(),
        "ssim": loss_ssim.detach(),
        "geometric": loss_g.detach(),
        "geometric_grad": loss_ggrad.detach(),
        "tgm": loss_tgm.detach(),
        "tv": loss_tv.detach(),
    }
