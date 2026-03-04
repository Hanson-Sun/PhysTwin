from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from da3_camera_config import CameraConfig


class ChunkProcessor:
    """Encapsulates the logic for processing a chunk of frames with the model."""
    
    def __init__(self, model):
        self.model = model
    
    def process_chunk(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str]
    ) -> Dict[int, List[np.ndarray]]:
        raise NotImplementedError("ChunkProcessor.process_chunk must be implemented by subclasses")

class PoseFinder:
    def __init__(self, model=None):
        self.model = model

    def infer_calibration(self, case_dir: Path, frame_names: List[str], calib_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Infer camera intrinsics and extrinsics from frames using the model."""
        raise NotImplementedError("PoseFinder.infer_calibration must be implemented by subclasses")

    def align_intrinsics(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Scale intrinsics to the depth-map resolution used by this model.

        Returns:
            (scaled_intrinsics, depth_hw) where depth_hw is (H, W), or (None, None) on failure.
        """
        raise NotImplementedError("PoseFinder.align_intrinsics must be implemented by subclasses")
    