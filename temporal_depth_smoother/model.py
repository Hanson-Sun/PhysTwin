"""
Temporal Depth Smoother: Dilated 3D CNN with U-Net skip connections.

Bidirectional temporal context via dilated 3D convolutions.
Dilation stack (1,2,4,8) gives ±15 frames receptive field.
U-Net encoder-decoder with skip connections preserves spatial detail.

Input:  depth_raw [B, T, H, W], rgb [B, T, H, W, 3]
Output: depth_smooth [B, T, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> tuple:
    """
    Pad spatial dimensions (last 2 dims) to nearest multiple of `multiple`.
    Returns (padded_tensor, original_shape) for later unpadding.
    
    Works with any layout as long as spatial dims are last two.
    """
    H, W = x.shape[-2], x.shape[-1]
    H_pad = ((H + multiple - 1) // multiple) * multiple
    W_pad = ((W + multiple - 1) // multiple) * multiple
    
    if H_pad == H and W_pad == W:
        return x, (H, W)
    
    padding = (0, W_pad - W, 0, H_pad - H)  # (left, right, top, bottom) for last 2 dims
    padded = F.pad(x, padding, mode='constant', value=0)
    return padded, (H, W)


def unpad(x: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Remove padding from tensor using original spatial dimensions.
    
    original_shape: (H, W) tuple from pad_to_multiple
    """
    H_orig, W_orig = original_shape
    return x[..., :H_orig, :W_orig]


class Conv2dBlock(nn.Module):
    """2D conv block applied per-frame."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DilatedTemporalBlock(nn.Module):
    """
    3D conv block with temporal dilation.
    Spatial dims use dilation=1 always; only time dim is dilated.
    """
    def __init__(self, ch: int, temporal_dilation: int):
        super().__init__()
        tp = temporal_dilation  # temporal padding = dilation for same-size output
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=(3,3,3),
                      padding=(tp,1,1), dilation=(temporal_dilation,1,1)),
            nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
        )
        # 1x1x1 residual projection if needed
        self.residual = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


class EncoderBlock(nn.Module):
    """Spatial encoder level: conv block + strided downsample."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = Conv2dBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        # x: [B*T, C, H, W]
        skip = self.conv(x)
        return self.down(skip), skip


class DecoderBlock(nn.Module):
    """Spatial decoder level: upsample + skip connection + conv block."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Conv2dBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TemporalDepthSmoother(nn.Module):
    """
    Architecture:
      1. Per-frame 2D encoding at 3 spatial scales (with skip connections saved)
      2. Dilated 3D conv stack at bottleneck: dilations (1,2,4,8) → ±15 frame receptive field
      3. Per-frame 2D decoding with U-Net skip connections
      4. Residual depth correction: depth_smooth = depth_raw + Δ
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        B  = config.base_channels
        Rd = config.rgb_encoder_channels
        Dd = config.depth_encoder_channels

        # Per-frame encoders (2D)
        self.rgb_enc   = Conv2dBlock(3, Rd)
        self.depth_enc = Conv2dBlock(1, Dd)

        # Spatial U-Net encoder (operates on concatenated rgb+depth features)
        in_ch = Rd + Dd
        self.enc1 = EncoderBlock(in_ch, B)        # full res → /2
        self.enc2 = EncoderBlock(B,     B * 2)    # /2 → /4
        self.enc3 = EncoderBlock(B * 2, B * 4)    # /4 → /8

        # Dilated 3D temporal bottleneck — dilation stack is a hyperparameter.
        # e.g. [1,2,4,8] → ±15 frame receptive field
        #      [1]        → ±1  frame (flicker only)
        #      [1,2]      → ±3  frames
        dilations = config.temporal_dilations
        rf = sum(dilations)  # approximate one-sided receptive field
        self.temporal = nn.Sequential(*[
            DilatedTemporalBlock(B * 4, d) for d in dilations
        ])
        print(f"Temporal stack: dilations={dilations}, receptive field ≈ ±{rf} frames")

        # Spatial U-Net decoder (2D, per-frame)
        self.dec3 = DecoderBlock(B * 4, B * 4, B * 2)   # /8 → /4
        self.dec2 = DecoderBlock(B * 2, B * 2, B)        # /4 → /2
        self.dec1 = DecoderBlock(B,     in_ch, B // 2)   # /2 → full res

        # Residual head
        self.head = nn.Conv2d(B // 2, 1, kernel_size=1)

    def forward(self, depth_raw: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_raw: [B, T, H, W]
            rgb:       [B, T, H, W, 3]
        Returns:
            depth_smooth: [B, T, H, W]
        
        Handles any input resolution by padding to nearest multiple of 8.
        """
        B, T, H, W = depth_raw.shape

        # Per-clip normalisation
        mean = depth_raw.mean(dim=[1,2,3], keepdim=True)
        std  = depth_raw.std( dim=[1,2,3], keepdim=True).clamp(min=1e-6)
        depth_norm = (depth_raw - mean) / std

        # Reformat RGB to [B, T, 3, H, W] so spatial dims are last 2
        rgb = rgb.permute(0, 1, 4, 2, 3)  # [B, T, 3, H, W]

        # Pad to multiple of 8 for stride requirement
        depth_norm_padded, orig_shape = pad_to_multiple(depth_norm, multiple=8)
        rgb_padded, _ = pad_to_multiple(rgb, multiple=8)
        
        H_pad, W_pad = depth_norm_padded.shape[-2:]

        # Flatten batch dim: [B, T, C, H_pad, W_pad] → [B*T, C, H_pad, W_pad]
        rgb_flat = rgb_padded.reshape(B*T, 3, H_pad, W_pad).float() / 255.0
        depth_flat = depth_norm_padded.reshape(B*T, 1, H_pad, W_pad)

        # ── Per-frame 2D feature extraction ──────────────────────────────
        feat0 = torch.cat([self.rgb_enc(rgb_flat),
                           self.depth_enc(depth_flat)], dim=1)  # [B*T, Rd+Dd, H, W]

        x2,   skip1 = self.enc1(feat0)   # skip1: [B*T, B,   H,   W  ]
        x4,   skip2 = self.enc2(x2)      # skip2: [B*T, B*2, H/2, W/2]
        x8,   skip3 = self.enc3(x4)      # skip3: [B*T, B*4, H/4, W/4]

        # ── Dilated 3D temporal reasoning ────────────────────────────────
        # Reshape to [B, C, T, H', W'] for 3D conv
        _, C8, H8, W8 = x8.shape
        x8_3d = x8.reshape(B, T, C8, H8, W8).permute(0, 2, 1, 3, 4)  # [B, C, T, H', W']
        x8_3d = self.temporal(x8_3d)
        x8    = x8_3d.permute(0, 2, 1, 3, 4).reshape(B*T, C8, H8, W8)

        # ── Per-frame 2D decoding with skip connections ───────────────────
        x = self.dec3(x8,   skip3)
        x = self.dec2(x,    skip2)
        x = self.dec1(x,    skip1)

        # ── Residual correction ───────────────────────────────────────────
        delta      = self.head(x).reshape(B, T, H_pad, W_pad)
        depth_norm_padded = depth_norm_padded + delta
        
        # Remove padding to restore original resolution
        depth_norm = unpad(depth_norm_padded, orig_shape)

        return depth_norm * std + mean