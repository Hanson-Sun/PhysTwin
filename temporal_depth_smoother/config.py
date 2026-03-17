"""Configuration and hyperparameters for Temporal Depth Smoother."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    T: int = 8  
    base_channels: int = 32  
    rgb_encoder_channels: int = 16  
    depth_encoder_channels: int = 16  
    temporal_dilations: List[int] = field(default_factory=lambda: [1, 2]) 


@dataclass
class TrainingConfig:
    batch_size: int = 1  
    learning_rate: float = 1e-3
    num_epochs: int = 50
    weight_decay: float = 1e-4
    lambda_temporal: float = 1.0
    lambda_geometric: float = 0.5
    lambda_smooth: float = 0.3
    motion_threshold: float = 0.05
    log_interval: int = 5
    use_amp: bool = True  # Automatic Mixed Precision for lower memory usage


@dataclass
class DataConfig:
    background_stddev_threshold: float = 0.01
    depth_std_clamp: float = 1e-6
    target_height: int = None
    target_width: int = None


@dataclass
class Config:
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None

    def __post_init__(self):
        if self.model    is None: self.model    = ModelConfig()
        if self.training is None: self.training = TrainingConfig()
        if self.data     is None: self.data     = DataConfig()


def get_default_config() -> Config:
    return Config()