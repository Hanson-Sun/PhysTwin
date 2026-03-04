from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from depth_inference_classes import PoseFinder


class DUSt3RPoseFinder(PoseFinder):
    """PoseFinder implementation for DUSt3R model.

    model may be None; if so, DUSt3R is loaded lazily on first call.
    """

    def infer_calibration(
        self,
        case_dir: Path,
        frame_names: List[str],
        calib_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        _sys.path.insert(0, str(Path(__file__).parent.parent / "dust3r"))
        from calibrate_with_dust3r import calibrate

        model = self.model
        if model is None:
            from dust3r.model import AsymmetricCroCo3DStereo
            model = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

        Ks, c2ws = calibrate(model, str(case_dir))
        intrinsics_array = np.stack(Ks).astype(np.float32)   # (N, 3, 3)
        c2w_array = np.stack(c2ws).astype(np.float32)        # (N, 4, 4)
        # DUSt3R returns cam-to-world; invert to world-to-camera.
        extrinsics_array = np.linalg.inv(c2w_array)          # (N, 4, 4)
        return intrinsics_array, extrinsics_array