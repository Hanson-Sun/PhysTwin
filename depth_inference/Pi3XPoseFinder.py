import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from depth_inference_classes import PoseFinder
from da3_camera_config import CameraConfig

_PIXEL_LIMIT = 255_000  # must match Pi3XChunkProcessor


class Pi3XPoseFinder(PoseFinder):
    """PoseFinder implementation for Pi3X model.

    Pi3X conditions depth inference on pre-calibrated poses and cannot
    infer poses from scratch.  Camera calibration is therefore delegated
    to DUSt3R.
    """

    def infer_calibration(
        self,
        case_dir: Path,
        frame_names: List[str],
        calib_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from DUSt3RPoseFinder import DUSt3RPoseFinder
        return DUSt3RPoseFinder().infer_calibration(case_dir, frame_names, calib_frames)

    def align_intrinsics(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Scale intrinsics to Pi3X's inference resolution (pixel-limit resize)."""
        import PIL.Image as PILImage

        frame_name = frame_names[0]
        first_img_path = str(camera_configs[0].image_dir / frame_name)
        try:
            first = PILImage.open(first_img_path).convert("RGB")
            W_orig, H_orig = first.size
            scale = math.sqrt(_PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
            k = round(W_orig * scale / 14)
            m = round(H_orig * scale / 14)
            while (k * 14) * (m * 14) > _PIXEL_LIMIT:
                if k / m > W_orig / H_orig:
                    k -= 1
                else:
                    m -= 1
            inf_W, inf_H = max(1, k) * 14, max(1, m) * 14
            scale_x = inf_W / W_orig
            scale_y = inf_H / H_orig
            scaled = []
            for cfg in camera_configs:
                K = cfg.intrinsics.copy().astype(np.float32)
                K[0] *= scale_x
                K[1] *= scale_y
                scaled.append(K)
            return np.stack(scaled), (inf_H, inf_W)
        except Exception as e:
            print(f"      ! Pi3XPoseFinder.align_intrinsics failed: {e}")
            return None, None