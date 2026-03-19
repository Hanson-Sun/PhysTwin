"""Inference utilities for temporal depth smoothing."""

import torch
import argparse
import numpy as np
from pathlib import Path

from .model import TemporalDepthSmoother
from .config import get_default_config


def sliding_window_inference(
    model: torch.nn.Module,
    depth: torch.Tensor,
    rgb: torch.Tensor,
    window_size: int = 16,
    overlap: int = 4,
) -> torch.Tensor:
    """
    Chunk-based inference for arbitrarily long videos.

    Splits [1, T, H, W] input into overlapping windows, runs the model on
    each, and stitches results by keeping only the centre frames of each
    window (the edges have less temporal context and are less reliable).

    Args:
        model:       trained TemporalDepthSmoother
        depth:       [1, T, H, W]
        rgb:         [1, T, H, W, 3]
        window_size: frames per chunk  (should match training T)
        overlap:     frames to discard from each edge of every chunk
                     must be < window_size // 2
    Returns:
        [1, T, H, W] smoothed depth
    """
    assert overlap < window_size // 2, "overlap must be less than window_size // 2"

    T = depth.shape[1]

    # Short video — process in one shot
    if T <= window_size:
        return model(depth, rgb)

    stride  = window_size - 2 * overlap
    output  = torch.zeros_like(depth)
    counts  = torch.zeros(T, device=depth.device)

    start = 0
    while start < T:
        end = min(start + window_size, T)

        # Pad last chunk if shorter than window_size
        chunk_d = depth[:, start:end]
        chunk_r = rgb[:, start:end]
        if end - start < window_size:
            pad = window_size - (end - start)
            chunk_d = torch.cat([chunk_d, chunk_d[:, -pad:]], dim=1)
            chunk_r = torch.cat([chunk_r, chunk_r[:, -pad:]], dim=1)

        out = model(chunk_d, chunk_r)  # [1, window_size, H, W]

        # Only keep centre frames — discard overlap on each side
        keep_start = overlap if start > 0        else 0
        keep_end   = window_size - overlap if end < T else window_size

        src = out[:, keep_start:keep_end]
        dst_start = start + keep_start
        dst_end   = dst_start + (keep_end - keep_start)
        dst_end   = min(dst_end, T)
        src = src[:, :dst_end - dst_start]

        output[:, dst_start:dst_end] += src
        counts[dst_start:dst_end]    += 1

        if end >= T:
            break
        start += stride

    # Average overlapping regions (counts should be 1 everywhere with correct overlap)
    output /= counts.view(1, T, 1, 1).clamp(min=1)
    return output


class DepthSmoother:
    """Wrapper for easy inference."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            print("Warning: no config in checkpoint, using defaults", flush=True)
            config = get_default_config().model

        self.model = TemporalDepthSmoother(config=config).to(self.device)
        
        state_dict = checkpoint['model_state']
        if all(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded model  dilations={config.temporal_dilations}  "
              f"device={self.device}", flush=True)

    def smooth(self,
               depth_raw: np.ndarray,
               rgb: np.ndarray,
               window_size: int = None,
               overlap: int = 4) -> np.ndarray:
        """
        Args:
            depth_raw:   [T, H, W]     float32, any scale
            rgb:         [T, H, W, 3]  uint8 or float [0, 1]
            window_size: frames per chunk. Defaults to None = full video in
                         one pass. Set to match training T for long videos
                         that don't fit in VRAM.
            overlap:     edge frames discarded per chunk — only used when
                         window_size is set. Should cover the model's
                         receptive field (sum of dilations).
        Returns:
            [T, H, W] smoothed depth, same scale as input
        """
        depth_t = torch.from_numpy(depth_raw.astype(np.float32))
        rgb_t   = torch.from_numpy(rgb.astype(np.float32))
        if rgb_t.max() > 1.0:
            rgb_t = rgb_t / 255.0

        depth_t = depth_t.unsqueeze(0).to(self.device)  # [1, T, H, W]
        rgb_t   = rgb_t.unsqueeze(0).to(self.device)    # [1, T, H, W, 3]

        T = depth_t.shape[1]
        window_size = window_size or T  # None → full video, no chunking

        n_chunks = 1 if T <= window_size else (T - 2*overlap) // (window_size - 2*overlap) + 1
        print(f"Smoothing {T} frames  window={window_size}  "
              f"{'(full video)' if window_size >= T else f'overlap={overlap}  {n_chunks} chunks'}",
              flush=True)

        with torch.no_grad():
            depth_smooth = sliding_window_inference(
                self.model, depth_t, rgb_t,
                window_size=window_size,
                overlap=overlap,
            )

        return depth_smooth.squeeze(0).cpu().numpy()  # [T, H, W]


def main():
    p = argparse.ArgumentParser(description='Smooth depth sequences')
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--depth-input',  required=True)
    p.add_argument('--rgb-input',    required=True)
    p.add_argument('--output',       required=True)
    p.add_argument('--device',       default='cuda')
    p.add_argument('--window-size',  type=int, default=16,
                   help='Frames per chunk — should match training temporal window')
    p.add_argument('--overlap',      type=int, default=4,
                   help='Edge frames discarded per chunk. Should cover receptive field.')
    args = p.parse_args()

    depth_raw = np.load(args.depth_input)
    rgb       = np.load(args.rgb_input)
    print(f"depth: {depth_raw.shape}  rgb: {rgb.shape}", flush=True)

    smoother     = DepthSmoother(args.checkpoint, device=args.device)
    depth_smooth = smoother.smooth(depth_raw, rgb,
                                   window_size=args.window_size,
                                   overlap=args.overlap)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, depth_smooth)
    print(f"Saved {args.output}  shape={depth_smooth.shape}", flush=True)


if __name__ == '__main__':
    main()