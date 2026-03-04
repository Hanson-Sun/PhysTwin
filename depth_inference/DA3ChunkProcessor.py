from depth_inference_classes import ChunkProcessor
import torch
from depth_anything_3.api import DepthAnything3
from da3_camera_config import CameraConfig, ensure_4x4_matrix
from typing import List, Dict
import numpy as np

class DA3ChunkProcessor(ChunkProcessor):
    """ChunkProcessor implementation for Depth Anything 3 model."""
    
    def __init__(self, model: DepthAnything3):
        super().__init__(model)

    def process_chunk(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str]
    ) -> Dict[int, List[np.ndarray]]:
        """
        Run depth inference on entire chunk at once with temporal consistency.

        Collects all frames' images for all cameras and passes to DA3 as a temporal batch
        to leverage both multi-view and temporal constraints simultaneously.
        """
        num_cams = len(camera_ids)
        num_frames = len(frame_names)
        
        # Collect image paths and camera parameters directly into flat arrays
        all_images = []
        all_intrinsics = []
        all_extrinsics = []
        
        for frame_name in frame_names:
            for config in camera_configs:
                all_images.append(str(config.image_dir / frame_name))
                all_intrinsics.append(config.intrinsics)
                all_extrinsics.append(config.extrinsics)
        
        # Convert to numpy arrays
        intrinsics_tensor = np.stack(all_intrinsics, axis=0)
        extrinsics_4x4 = [ensure_4x4_matrix(ext) for ext in all_extrinsics]
        extrinsics_tensor = np.stack(extrinsics_4x4, axis=0)
        
        # Run inference on entire chunk with temporal + multi-view context
        with torch.no_grad():
            prediction = self.model.inference(
                image=all_images,
                intrinsics=intrinsics_tensor,  
                extrinsics=extrinsics_tensor, 
                # use_ray_pose=True,
                align_to_input_ext_scale=True
            )
        
        if prediction.depth is None:
            raise RuntimeError("DA3 failed to infer depth")

        num_frames = len(frame_names)
        num_cams = len(camera_ids)
        chunk_depths = {cam_id: [] for cam_id in camera_ids}

        for frame_idx in range(num_frames):
            for cam_idx, cam_id in enumerate(camera_ids):
                depth = prediction.depth[frame_idx * num_cams + cam_idx]
                if torch.is_tensor(depth):
                    depth = depth.detach().cpu().numpy()
                chunk_depths[cam_id].append(depth)
        
        del prediction, all_images, all_intrinsics, all_extrinsics
        del intrinsics_tensor, extrinsics_tensor, extrinsics_4x4
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 

        return chunk_depths
