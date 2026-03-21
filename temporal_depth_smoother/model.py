"""
Temporal Depth Smoother: Dilated 3D CNN with U-Net skip connections.

Bidirectional temporal context via dilated 3D convolutions.
Dilation stack (1,2,4,8) gives ±15 frames receptive field.
U-Net encoder-decoder with skip connections preserves spatial detail.

Multi-scale temporal processing (aligned with MED-VT, StableDPT):
  - temporal        : dilated stack at /8  — long-range context (±15 frames)
  - temporal_mid    : dilation=1 at /4    — medium-scale boundary precision
  - temporal_shallow: dilation=1 at /2    — fine-grained pixel-level smoothing

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

    padding = (0, W_pad - W, 0, H_pad - H)  # (left, right, top, bottom)
    padded = F.pad(x, padding, mode='constant', value=0)
    return padded, (H, W)


def unpad(x: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Remove padding using original spatial dimensions.

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
    Spatial dims always use dilation=1; only the time dim is dilated.
    Includes a residual connection for stable gradient flow.
    """
    def __init__(self, ch: int, temporal_dilation: int):
        super().__init__()
        tp = temporal_dilation  # temporal padding = dilation → same-size output
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=(3, 3, 3),
                      padding=(tp, 1, 1), dilation=(temporal_dilation, 1, 1)),
            nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
        )
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
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TemporalDepthSmoother(nn.Module):
    """
    Architecture (coarse-to-fine temporal processing, per MED-VT / StableDPT):

      1. Per-frame 2D encoding at 3 spatial scales → skip1 (/2), skip2 (/4), skip3 (/8)

      2. Multi-scale temporal blocks applied to every skip before decoding:
           temporal        : dilated [1,2,4,8] on bottleneck (/8) — long-range context
           temporal_mid    : dilation=1 on skip2 (/4)             — medium-scale precision
           temporal_shallow: dilation=1 on skip1 (/2)             — fine pixel-level smoothing
         skip3 is implicitly covered by the bottleneck block sitting directly above it.

      3. Per-frame 2D decoding with temporally-aware skip connections at all scales.

      4. Residual depth correction: depth_smooth = depth_raw + Δ
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.config = config

        B  = config.base_channels
        Rd = config.rgb_encoder_channels
        Dd = config.depth_encoder_channels

        # Per-frame encoders (2D)
        self.rgb_enc   = Conv2dBlock(3, Rd)
        self.depth_enc = Conv2dBlock(1, Dd)

        # Spatial U-Net encoder 
        in_ch = Rd + Dd
        self.enc1 = EncoderBlock(in_ch, B)        # full res → /2   skip1: B ch
        self.enc2 = EncoderBlock(B,     B * 2)    # /2       → /4   skip2: B*2 ch
        self.enc3 = EncoderBlock(B * 2, B * 4)    # /4       → /8   skip3: B*4 ch

        # Temporal blocks (multi-scale) 
        # Deep: dilated stack at /8 for long-range temporal context.
        dilations = config.temporal_dilations
        rf = sum(dilations)
        self.temporal = nn.Sequential(*[
            DilatedTemporalBlock(B * 4, d) for d in dilations
        ])
        print(f"Temporal stack: dilations={dilations}, receptive field ≈ ±{rf} frames")

        # Mid: dilation=1 at /4. Gives dec2 temporally-aware features at medium
        # spatial scale, preventing coarse block artefacts at object boundaries.
        # Cheap — operates at H/4 which is 4× smaller area than H/2.
        self.temporal_mid = DilatedTemporalBlock(B * 2, temporal_dilation=1)

        # Shallow: dilation=1 at /2. Finest-grained temporal pass — ensures
        # dec1 makes sharp per-pixel smoothing decisions rather than upsampling
        # blobs from the coarser temporal block.
        self.temporal_shallow = DilatedTemporalBlock(B, temporal_dilation=1)

        # Spatial U-Net decoder 
        self.dec3 = DecoderBlock(B * 4, B * 4, B * 2)   # /8 → /4
        self.dec2 = DecoderBlock(B * 2, B * 2, B)        # /4 → /2
        self.dec1 = DecoderBlock(B,     B,     B // 2)   # /2 → full res

        # Residual head 
        self.head = nn.Conv2d(B // 2, 1, kernel_size=1)

    def _apply_temporal(self, feat: torch.Tensor, block: nn.Module,
                        B: int, T: int) -> torch.Tensor:
        """
        Reshape [B*T, C, H, W] → [B, C, T, H, W], apply temporal block,
        reshape back to [B*T, C, H, W].
        """
        BT, C, H, W = feat.shape
        x3d = feat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x3d = block(x3d)
        return x3d.permute(0, 2, 1, 3, 4).reshape(BT, C, H, W)

    def forward(self, depth_raw: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_raw: [B, T, H, W]
            rgb:       [B, T, H, W, 3]  — uint8 or float [0, 1]
        Returns:
            depth_smooth: [B, T, H, W]

        Handles any input resolution by padding to nearest multiple of 8.
        """
        B, T, H, W = depth_raw.shape

        # Per-clip normalisation 
        mean = depth_raw.mean(dim=[1, 2, 3], keepdim=True)
        std  = depth_raw.std( dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        depth_norm = (depth_raw - mean) / std

        # Reformat + pad 
        rgb = rgb.permute(0, 1, 4, 2, 3)                          # [B, T, 3, H, W]
        depth_norm_padded, orig_shape = pad_to_multiple(depth_norm, multiple=8)
        rgb_padded, _                 = pad_to_multiple(rgb,        multiple=8)
        H_pad, W_pad = depth_norm_padded.shape[-2:]

        # Flatten to [B*T, C, H, W]
        rgb_flat = rgb_padded.reshape(B * T, 3, H_pad, W_pad).float()
        if rgb_flat.max() > 1.0:
            rgb_flat = rgb_flat / 255.0
        depth_flat = depth_norm_padded.reshape(B * T, 1, H_pad, W_pad)

        # Per-frame 2D feature extraction 
        feat0 = torch.cat([self.rgb_enc(rgb_flat),
                           self.depth_enc(depth_flat)], dim=1)  # [B*T, Rd+Dd, H, W]

        x2, skip1 = self.enc1(feat0)   # skip1: [B*T, B,   H/2, W/2]
        x4, skip2 = self.enc2(x2)      # skip2: [B*T, B*2, H/4, W/4]
        x8, skip3 = self.enc3(x4)      # skip3: [B*T, B*4, H/8, W/8]

        # Deep temporal at /8 — long-range context 
        x8 = self._apply_temporal(x8, self.temporal, B, T)

        # Mid temporal on skip2 at /4 — medium-scale precision 
        skip2 = self._apply_temporal(skip2, self.temporal_mid, B, T)

        # Shallow temporal on skip1 at /2 — fine pixel-level smoothing 
        skip1 = self._apply_temporal(skip1, self.temporal_shallow, B, T)

        # Per-frame 2D decoding 
        # skip3 enters dec3 without a separate temporal block — it is already
        # implicitly temporally processed because it was computed directly before
        # the bottleneck temporal block, and the bottleneck output x8 that dec3
        # receives has attended to it at /8 resolution.
        x = self.dec3(x8,   skip3)
        x = self.dec2(x,    skip2)
        x = self.dec1(x,    skip1)

        # Residual correction
        delta             = self.head(x).reshape(B, T, H_pad, W_pad)
        depth_norm_padded = depth_norm_padded + delta
        depth_norm        = unpad(depth_norm_padded, orig_shape)

        return depth_norm * std + mean