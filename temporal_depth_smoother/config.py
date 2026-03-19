"""Configuration and hyperparameters for Temporal Depth Smoother."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    T: int = 16
    base_channels: int = 48
    rgb_encoder_channels: int = 16
    depth_encoder_channels: int = 16
    temporal_dilations: List[int] = field(default_factory=lambda: [1, 2, 4])


@dataclass
class TrainingConfig:
    batch_size: int = 2
    learning_rate: float = 3e-3
    num_epochs: int = 50
    weight_decay: float = 1e-4
    lambda_tgm: float = 1.50
    lambda_fidelity: float = 1.0
    lambda_tv: float = 1.0
    lambda_geometric: float = 0.25    
    lambda_flicker: float = 0.0
    log_interval: int = 10
    use_amp: bool = True


@dataclass
class DataConfig:
    background_stddev_threshold: float = 0.01
    depth_std_clamp: float = 1e-6
    target_height: int = 216
    target_width: int = 384


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