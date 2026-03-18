"""Data loading for temporal depth smoothing."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List


def scan_clips(data_dir: Path) -> list:
    """Find all clips with complete data files."""
    clips = []
    for depth_file in sorted(data_dir.glob('*_depth_raw.npy')):
        clip_id = depth_file.stem.replace('_depth_raw', '')
        if (data_dir / f"{clip_id}_rgb.npy").exists() and \
           (data_dir / f"{clip_id}_depth_vda_aligned.npy").exists():
            clips.append(clip_id)
        else:
            print(f"  Warning: incomplete files for {clip_id}, skipping", flush=True)
    return clips


def split_clips(clips: list, val_fraction: float = 0.2, seed: int = 42) -> Tuple[list, list]:
    """
    Deterministic clip-level train/val split.
    Split is done at clip level so no frames from a val clip appear in training.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips))
    n_val = max(1, int(len(clips) * val_fraction))
    val_idx     = set(idx[:n_val].tolist())
    train_clips = [c for i, c in enumerate(clips) if i not in val_idx]
    val_clips   = [c for i, c in enumerate(clips) if i in val_idx]
    return train_clips, val_clips


def variable_size_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-sized batches.
    Pads all tensors to max size within this batch.
    """
    keys = batch[0].keys()
    result = {}

    for key in keys:
        tensors = [item[key] for item in batch]
        shapes  = [t.shape for t in tensors]

        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))

        padded = []
        for t in tensors:
            if t.shape != max_shape:
                padding = []
                for i in range(len(t.shape) - 1, -1, -1):
                    padding.extend([0, max_shape[i] - t.shape[i]])
                t = F.pad(t, padding, mode='constant', value=0)
            padded.append(t)

        result[key] = torch.stack(padded, dim=0)

    return result


class TemporalDepthDataset(Dataset):
    """
    Sliding window dataset over depth/rgb/vda clips.

    Each sample is a [temporal_window, H, W] chunk extracted from a clip.
    Stride controls overlap between windows:
      - stride < temporal_window  → overlapping windows (more samples, more augmentation)
      - stride = temporal_window  → non-overlapping windows (fewer samples, no overlap)

    For training, stride = temporal_window // 2 gives 50% overlap.
    For validation, stride = temporal_window gives non-overlapping windows.
    """

    def __init__(self,
                 data_dir: str,
                 clips: list,
                 temporal_window: int = 16,
                 stride: int = None,
                 target_height: int = None,
                 target_width: int = None,
                 max_samples: Optional[int] = None):
        self.data_dir        = Path(data_dir)
        self.temporal_window = temporal_window
        self.stride          = stride if stride is not None else temporal_window
        self.target_height   = target_height
        self.target_width    = target_width

        self.samples = self._generate_samples(clips)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  {len(clips)} clips → {len(self.samples)} samples "
              f"(window={temporal_window}, stride={self.stride})", flush=True)

    def _generate_samples(self, clips: list) -> list:
        samples = []
        for clip_id in clips:
            try:
                T = np.load(self.data_dir / f"{clip_id}_depth_raw.npy",
                            mmap_mode='r').shape[0]
                # Use self.stride — this was the bug before (was hardcoded to temporal_window)
                for start in range(0, T - self.temporal_window + 1, self.stride):
                    samples.append((clip_id, start))
            except Exception as e:
                print(f"  Warning: skipping {clip_id}: {e}", flush=True)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_mmap(self, clip_id: str):
        """
        Return mmap handles for a clip, opening them lazily on first access.
        mmap_mode='r' means only the requested slice is read from disk.
        Worker-safe: each DataLoader worker gets its own forked copy.
        """
        if not hasattr(self, '_mmaps'):
            self._mmaps = {}
        if clip_id not in self._mmaps:
            self._mmaps[clip_id] = {
                'depth_raw': np.load(self.data_dir / f"{clip_id}_depth_raw.npy",          mmap_mode='r'),
                'rgb':       np.load(self.data_dir / f"{clip_id}_rgb.npy",                mmap_mode='r'),
                'depth_vda': np.load(self.data_dir / f"{clip_id}_depth_vda_aligned.npy",  mmap_mode='r'),
            }
        return self._mmaps[clip_id]

    def _resize(self, depth: torch.Tensor, is_rgb: bool = False) -> torch.Tensor:
        """
        Resize spatial dimensions to target_height x target_width if set.
        depth: [T, H, W] or [T, H, W, 3]
        """
        if self.target_height is None or self.target_width is None:
            return depth

        if is_rgb:
            # [T, H, W, 3] → [T, 3, H, W] for interpolate → back
            x = depth.permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(self.target_height, self.target_width),
                              mode='bilinear', align_corners=False)
            return x.permute(0, 2, 3, 1)
        else:
            # [T, H, W] → [T, 1, H, W] for interpolate → back
            x = depth.unsqueeze(1).float()
            x = F.interpolate(x, size=(self.target_height, self.target_width),
                              mode='bilinear', align_corners=False)
            return x.squeeze(1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_id, start = self.samples[idx]
        end = start + self.temporal_window

        mmaps     = self._get_mmap(clip_id)
        depth_raw = torch.from_numpy(mmaps['depth_raw'][start:end].astype(np.float32))
        rgb       = torch.from_numpy(mmaps['rgb'][start:end].astype(np.float32))
        depth_vda = torch.from_numpy(mmaps['depth_vda'][start:end].astype(np.float32))

        # Normalize RGB to [0, 1]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        # Resize if target resolution is set
        depth_raw = self._resize(depth_raw, is_rgb=False)
        depth_vda = self._resize(depth_vda, is_rgb=False)
        rgb       = self._resize(rgb,       is_rgb=True)

        return {
            'depth_raw':         depth_raw,
            'rgb':               rgb,
            'depth_vda_aligned': depth_vda,
        }


def create_dataloaders(data_dir: str,
                       batch_size: int = 4,
                       temporal_window: int = 16,
                       val_fraction: float = 0.2,
                       num_workers: int = 2,
                       pin_memory: bool = True,
                       target_height: int = None,
                       target_width: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    Clip-level train/val split — no val clip frames appear in training.
    """
    data_dir = Path(data_dir)
    clips    = scan_clips(data_dir)
    assert clips, f"No complete clips found in {data_dir}"

    train_clips, val_clips = split_clips(clips, val_fraction=val_fraction)
    print(f"Split: {len(train_clips)} train clips, {len(val_clips)} val clips", flush=True)

    train_dataset = TemporalDepthDataset(
        data_dir, train_clips,
        temporal_window=temporal_window,
        stride=temporal_window // 2,
        target_height=target_height,
        target_width=target_width,
    )
    val_dataset = TemporalDepthDataset(
        data_dir, val_clips,
        temporal_window=temporal_window,
        stride=temporal_window,   
        target_height=target_height,
        target_width=target_width,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=variable_size_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=variable_size_collate)

    return train_loader, val_loader