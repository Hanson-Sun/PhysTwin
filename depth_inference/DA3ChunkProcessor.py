from typing import Dict, List

import numpy as np
import torch
from da3_camera_config import CameraConfig, ensure_4x4_matrix
from depth_anything_3.api import DepthAnything3
from depth_inference_classes import ChunkProcessor


class DA3ChunkProcessor(ChunkProcessor):
    """ChunkProcessor implementation for Depth Anything 3."""

    def __init__(self, model: DepthAnything3):
        super().__init__(model)

    def process_chunk(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str],
    ) -> Dict[int, List[np.ndarray]]:
        """Run DA3 inference on all frames × cameras as a single temporal batch."""
        num_cams = len(camera_ids)
        num_frames = len(frame_names)

        all_images, all_intrinsics, all_extrinsics = [], [], []
        for frame_name in frame_names:
            for config in camera_configs:
                all_images.append(str(config.image_dir / frame_name))
                all_intrinsics.append(config.intrinsics)
                all_extrinsics.append(config.extrinsics)

        intrinsics_arr = np.stack(all_intrinsics, axis=0)
        extrinsics_arr = np.stack(
            [ensure_4x4_matrix(ext) for ext in all_extrinsics], axis=0
        )

        with torch.no_grad():
            prediction = self.model.inference(
                image=all_images,
                intrinsics=intrinsics_arr,
                extrinsics=extrinsics_arr,
                align_to_input_ext_scale=True,
            )

        if prediction.depth is None:
            raise RuntimeError("DA3 failed to infer depth")

        chunk_depths: Dict[int, List[np.ndarray]] = {
            cam_id: [] for cam_id in camera_ids
        }
        for frame_idx in range(num_frames):
            for cam_idx, cam_id in enumerate(camera_ids):
                depth = prediction.depth[frame_idx * num_cams + cam_idx]
                if torch.is_tensor(depth):
                    depth = depth.detach().cpu().numpy()
                chunk_depths[cam_id].append(depth)

        del prediction, all_images, all_intrinsics, all_extrinsics
        del intrinsics_arr, extrinsics_arr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return chunk_depths
