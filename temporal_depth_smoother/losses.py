"""Loss functions for temporal depth smoothing."""

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_SOBEL_Y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_sobel_cache: dict = {}


def _get_sobel(device):
    if device not in _sobel_cache:
        _sobel_cache[device] = (_SOBEL_X.to(device), _SOBEL_Y.to(device))
    return _sobel_cache[device]


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
    sx, sy = _get_sobel(gray.device)
    grad = (F.conv2d(gray, sx, padding=1) ** 2 +
            F.conv2d(gray, sy, padding=1) ** 2).sqrt().squeeze(1)
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


def compute_flicker_mask(depth_raw: torch.Tensor,
                         depth_vda: torch.Tensor,
                         rgb: torch.Tensor,
                         depth_threshold: float = 0.15,
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

def _tgm_baseline(depth_raw: torch.Tensor,
                  depth_vda: torch.Tensor,
                  k: int = 1) -> torch.Tensor:
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


def _tv_baseline(depth_raw: torch.Tensor,
                 depth_vda: torch.Tensor, 
                 k: int = 1) -> torch.Tensor:
    """
    TV at identity = DA3's own second derivative, gated by VDA's acceleration.
    Uses DA3 (not VDA) because VDA is very smooth — its second derivative is
    near zero, which would inflate the normalised TV loss to very large values.
    """
    with torch.no_grad():
        d2_raw = depth_raw[:, 2*k:] - 2 * depth_raw[:, k:-k] + depth_raw[:, :-2*k]
        d2_vda = (depth_vda[:, 2*k:] - 2 * depth_vda[:, k:-k] + depth_vda[:, :-2*k]).abs()
        weight = torch.exp(-10.0 * d2_vda)
        return d2_raw.abs().mul(weight).mean().detach().clamp(min=1e-6)


def _fidelity_baseline(depth_raw: torch.Tensor,
                       t: torch.Tensor,
                       s: torch.Tensor) -> torch.Tensor:
    """
    Fidelity identity residual = 0, so we use the std of the aligned raw
    depth as the scale reference. A fidelity loss of 1.0 means the output
    deviates from DA3 by one standard deviation in aligned space.
    """
    with torch.no_grad():
        raw_aligned = (depth_raw - t.detach()) / s.detach()
        return raw_aligned.std().detach().clamp(min=1e-6)



def l_fidelity(depth_smooth: torch.Tensor,
               depth_raw: torch.Tensor,
               depth_vda: torch.Tensor,
               gate_floor: float = 0.25) -> torch.Tensor:
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

    # Clip-level anchor: temporal median of DA3 is robust to per-frame flicker
    raw_median = depth_raw.median(dim=1, keepdim=True).values      # [B, 1, H, W]
    flat       = raw_median.reshape(B, -1)
    t          = flat.median(dim=-1).values.view(B, 1, 1, 1)
    s          = (flat - t.reshape(B, 1)).abs().mean(dim=-1).view(B, 1, 1, 1).clamp(min=1e-6)

    smooth_aligned = (depth_smooth - t.detach()) / s.detach()
    raw_aligned    = (depth_raw    - t.detach()) / s.detach()
    residual       = torch.abs(smooth_aligned - raw_aligned)        # [B, T, H, W]

    # Gate: 1 on stable pixels, fades to gate_floor where VDA moves
    grad_vda      = (depth_vda[:, 1:] - depth_vda[:, :-1]).abs()   # [B, T-1, H, W]
    grad_vda_norm = grad_vda / (grad_vda.mean() + 1e-6)
    gate_pairs    = gate_floor + (1.0 - gate_floor) * torch.exp(-grad_vda_norm)
    gate_frame0   = torch.ones(B, 1, H, W, device=depth_smooth.device)
    gate          = torch.cat([gate_frame0, gate_pairs], dim=1)     # [B, T, H, W]

    raw_loss = residual.mul(gate).mean()
    baseline = _fidelity_baseline(depth_raw, t, s)
    return raw_loss / baseline


def l_flicker(depth_smooth: torch.Tensor,
              flicker_mask: torch.Tensor) -> torch.Tensor:
    """
    Suppress depth changes on pixels identified as flickering.
    Not normalised — only active when lambda_flicker > 0.
    """
    change = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return change.mul(flicker_mask).mean()


def l_tgm_k(depth_smooth: torch.Tensor, depth_vda: torch.Tensor, depth_raw: torch.Tensor, k: int = 1) -> torch.Tensor:
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
    grad_vda    = depth_vda[:, k:]    - depth_vda[:, :-k]
    raw_loss    = torch.abs(grad_smooth - grad_vda).mean()

    baseline = _tgm_baseline(depth_raw, depth_vda, k) 
    return raw_loss / baseline

def l_tv_k(depth_smooth: torch.Tensor, depth_vda: torch.Tensor, depth_raw: torch.Tensor, k: int = 1) -> torch.Tensor:
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

    baseline = _tv_baseline(depth_raw, depth_vda, k=k)
    return raw_loss / baseline




def l_geometric(depth_smooth: torch.Tensor,
                depth_raw: torch.Tensor,
                rgb: torch.Tensor,
                edge_weight_strength: float = 5.0) -> torch.Tensor:
    """
    Edge-focused spatial structure loss.

    Enforces that depth edges in the output match DA3's depth edges,
    weighted by RGB edge strength. Pixels at strong RGB edges (object
    boundaries) get high weight -- these are exactly where smearing occurs
    and where DA3's spatial gradients are most meaningful. Flat regions 
    get near-zero weight.

    Args:
        depth_smooth:         [B, T, H, W]
        depth_raw:            [B, T, H, W]
        rgb:                  [B, T, H, W, 3]  float [0, 1]
        edge_weight_strength: higher = more focus on strong edges only
    Returns:
        scalar loss (unnormalised)
    """
    smooth_gx, smooth_gy = compute_depth_spatial_gradient(depth_smooth)
    raw_gx,    raw_gy    = compute_depth_spatial_gradient(depth_raw)

    edge_w = compute_image_gradient(rgb)                           # [B, T, H, W]
    edge_w = edge_w / (edge_w.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
    edge_w = edge_w.clamp(min=0.1)

    loss_x = (smooth_gx - raw_gx).abs().mul(edge_w).mean()
    loss_y = (smooth_gy - raw_gy).abs().mul(edge_w).mean()
    return (loss_x + loss_y) / 2.0


# deprecated
def l_smooth(depth_smooth: torch.Tensor,
             motion_mask: torch.Tensor) -> torch.Tensor:
    """Deprecated — set lambda_smooth=0."""
    grad = (depth_smooth[:, 1:] - depth_smooth[:, :-1]).abs()
    return grad.mul(1 - motion_mask).mean()


# ── Total loss ─────────────────────────────────────────────────────────────────

def total_loss(depth_smooth: torch.Tensor,
               depth_raw: torch.Tensor,
               depth_vda_aligned: torch.Tensor,
               rgb: torch.Tensor,
               lambda_fidelity: float = 1.0,
               lambda_tgm: float = 1.0,
               lambda_geometric: float = 1.0,
               lambda_tv: float = 1.0,
               lambda_flicker: float = 0.0,
               depth_threshold: float = 0.15,
               rgb_threshold: float = 0.05,
               vda_weight: float = 0.7) -> dict:
    """
    fidelity, tgm, and tv are normalised to ~1.0 at the identity baseline.
    geometric is NOT normalised (its identity value = 0, baseline is unstable)
    — use a small lambda_geometric (0.01-0.02) to keep it comparable.

    Lambda meaning for normalised losses: lambda=1.0 means this loss
    contributes one "identity unit" of gradient signal.

    Args:
        depth_smooth:      [B, T, H, W]  model output
        depth_raw:         [B, T, H, W]  DA3 input
        depth_vda_aligned: [B, T, H, W]  VDA aligned depth
        rgb:               [B, T, H, W, 3]  uint8 or float [0, 1]
        lambda_fidelity:   geometry preservation weight
        lambda_tgm:        temporal gradient matching weight
        lambda_geometric:  spatial structure weight — keep small (0.01-0.02)
        lambda_tv:         second-order temporal smoothness weight
        lambda_flicker:    flicker mask suppression weight (0 = disabled)
    Returns:
        dict: 'total', 'fidelity', 'tgm', 'geometric', 'tv', 'flicker'
    """
    rgb_01 = rgb.float().div(255.0) if rgb.max() > 1.0 else rgb.float()

    loss_fid = l_fidelity(depth_smooth, depth_raw, depth_vda_aligned)
    loss_g   = l_geometric(depth_smooth, depth_raw, rgb_01)
    loss_tv_1  = l_tv_k(depth_smooth, depth_vda_aligned, depth_raw, k=1)
    loss_tv_2  = l_tv_k(depth_smooth, depth_vda_aligned, depth_raw, k=2)
    # loss_tv_4  = l_tv_k(depth_smooth, depth_vda_aligned, depth_raw, k=4)
    loss_tv = loss_tv_1 + 0.2 * loss_tv_2 
    loss_tgm_4 = l_tgm_k(depth_smooth, depth_vda_aligned, depth_raw, k=3)
    loss_tgm_2 = l_tgm_k(depth_smooth, depth_vda_aligned, depth_raw, k=2)
    loss_tgm_1 = l_tgm_k(depth_smooth, depth_vda_aligned, depth_raw, k=1)
    loss_tgm = loss_tgm_1 + 0.2 * loss_tgm_2 + 0.05 * loss_tgm_4

    loss_f = torch.tensor(0.0, device=depth_smooth.device)
    if lambda_flicker > 0.0:
        flicker_mask = compute_flicker_mask(
            depth_raw, depth_vda_aligned, rgb_01,
            depth_threshold=depth_threshold,
            rgb_threshold=rgb_threshold,
            vda_weight=vda_weight,
        )
        loss_f = l_flicker(depth_smooth, flicker_mask)

    total = (lambda_fidelity  * loss_fid +
             lambda_tgm       * loss_tgm +
             lambda_geometric * loss_g   +
             lambda_tv        * loss_tv  +
             lambda_flicker   * loss_f)

    return {
        'total':     total,
        'fidelity':  loss_fid.detach(),
        'tgm':       loss_tgm.detach(),
        'geometric': loss_g.detach(),
        'tv':        loss_tv.detach(),
        'flicker':   loss_f.detach(),
    }