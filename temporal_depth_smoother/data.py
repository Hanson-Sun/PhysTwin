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
    val_idx   = set(idx[:n_val].tolist())
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
        shapes = [t.shape for t in tensors]
        
        # Find max dimensions for this batch
        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
        
        # Pad all to max and stack
        padded = []
        for t in tensors:
            if t.shape != max_shape:
                # Create padding for this tensor
                padding = []
                for i in range(len(t.shape) - 1, -1, -1):
                    pad_amount = max_shape[i] - t.shape[i]
                    padding.extend([0, pad_amount])
                t = F.pad(t, padding, mode='constant', value=0)
            padded.append(t)
        
        result[key] = torch.stack(padded, dim=0)
    
    return result


class TemporalDepthDataset(Dataset):
    """
    Sliding window dataset over depth/rgb/vda clips.

    Each sample is a [temporal_window, H, W] chunk extracted from a clip.
    Stride controls overlap: temporal_window//2 for train, temporal_window for val.
    """

    def __init__(self,
                 data_dir: str,
                 clips: list,
                 temporal_window: int = 16,
                 stride: int = 8,
                 target_height: int = None,
                 target_width: int = None,
                 max_samples: Optional[int] = None):
        self.data_dir       = Path(data_dir)
        self.temporal_window = temporal_window

        self.samples = self._generate_samples(clips)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  {len(clips)} clips → {len(self.samples)} samples "
              f"(window={temporal_window}, stride={stride}, variable resolution)", flush=True)

    def _generate_samples(self, clips: list) -> list:
        samples = []
        for clip_id in clips:
            try:
                T = np.load(self.data_dir / f"{clip_id}_depth_raw.npy",
                            mmap_mode='r').shape[0]
                for start in range(0, T - self.temporal_window + 1, self.temporal_window):
                    samples.append((clip_id, start))
            except Exception as e:
                print(f"  Warning: skipping {clip_id}: {e}", flush=True)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_mmap(self, clip_id: str):
        """
        Return mmap handles for a clip, opening them lazily on first access.
        mmap_mode='r' means only the requested slice is read from disk —
        the full array is never loaded into RAM.
        Worker-safe: each DataLoader worker gets its own copy of the dict
        since Dataset is forked per worker.
        """
        if not hasattr(self, '_mmaps'):
            self._mmaps = {}
        if clip_id not in self._mmaps:
            self._mmaps[clip_id] = {
                'depth_raw': np.load(self.data_dir / f"{clip_id}_depth_raw.npy",       mmap_mode='r'),
                'rgb':       np.load(self.data_dir / f"{clip_id}_rgb.npy",             mmap_mode='r'),
                'depth_vda': np.load(self.data_dir / f"{clip_id}_depth_vda_aligned.npy", mmap_mode='r'),
            }
        return self._mmaps[clip_id]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_id, start = self.samples[idx]
        end = start + self.temporal_window

        mmaps = self._get_mmap(clip_id)
        depth_raw = torch.from_numpy(mmaps['depth_raw'][start:end].astype(np.float32))
        rgb       = torch.from_numpy(mmaps['rgb'][start:end].astype(np.float32))
        depth_vda = torch.from_numpy(mmaps['depth_vda'][start:end].astype(np.float32))

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

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
                       target_height: int = 434,
                       target_width: int = 756) -> Tuple[DataLoader, DataLoader]:
    """
    Clip-level train/val split — no val clip frames appear in training.

    Args:
        val_fraction: fraction of clips held out for validation
    """
    data_dir = Path(data_dir)
    clips = scan_clips(data_dir)
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
