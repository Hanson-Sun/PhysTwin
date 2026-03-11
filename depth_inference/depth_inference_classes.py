from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from da3_camera_config import CameraConfig


class ChunkProcessor(ABC):
    """Abstract base class for chunk-based depth inference."""

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def process_chunk(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str],
    ) -> Dict[int, List[np.ndarray]]:
        """Run depth inference on a chunk of frames.

        Returns:
            Dict mapping camera_id → list of depth maps (one per frame).
        """


class PoseFinder(ABC):
    """Abstract base class for camera pose / calibration inference."""

    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def infer_calibration(
        self,
        case_dir: Path,
        frame_names: List[str],
        calib_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer camera calibration from a set of frames.

        Returns:
            (intrinsics, extrinsics) as arrays of shape (N, 3, 3) and (N, 4, 4).
            Extrinsics are world-to-camera matrices.
        """

    @abstractmethod
    def align_intrinsics(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Return intrinsics scaled to the depth-model's output resolution.

        Returns:
            (scaled_intrinsics, depth_hw) where depth_hw is (H, W),
            or (None, None) on failure.
        """
