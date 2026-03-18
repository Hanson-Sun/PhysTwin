"""Temporal Depth Smoother: System-agnostic post-processing for temporal depth consistency."""

from .model import TemporalDepthSmoother
from .config import Config, get_default_config, ModelConfig, TrainingConfig, DataConfig
from .losses import (
    total_loss,
    l_flicker,
    l_geometric,
    l_smooth,
    compute_motion_mask,
)
from .data import (
    TemporalDepthDataset,
    create_dataloaders,
)
from .utils import (
    normalize_depth_clip,
    denormalize_depth,
    align_vda_to_da3,
    sliding_window_inference,
    compute_temporal_variance,
    save_checkpoint,
    load_checkpoint,
)
from .inference import DepthSmoother

__version__ = '0.1.0'

__all__ = [
    'TemporalDepthSmoother',
    'Config',
    'get_default_config',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'total_loss',
    'l_flicker',
    'l_geometric',
    'l_smooth',
    'compute_motion_mask',
    'TemporalDepthDataset',
    'create_dataloaders',
    'normalize_depth_clip',
    'denormalize_depth',
    'align_vda_to_da3',
    'sliding_window_inference',
    'compute_temporal_variance',
    'save_checkpoint',
    'load_checkpoint',
    'DepthSmoother',
]
