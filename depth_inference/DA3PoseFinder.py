from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from depth_inference_classes import PoseFinder
from da3_camera_config import CameraConfig, ensure_4x4_matrix, average_intrinsics, average_extrinsics


class DA3PoseFinder(PoseFinder):
    """PoseFinder implementation for Depth Anything 3 model."""

    def infer_calibration(
        self,
        case_dir: Path,
        frame_names: List[str],
        calib_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_calib_frames = max(1, min(calib_frames, len(frame_names)))
        print(f"\n[*] Inferring camera calibration by averaging over {num_calib_frames} frame(s)...")

        color_dir = case_dir / "color"
        camera_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
        num_cams = len(camera_dirs)

        if not frame_names:
            raise ValueError("No frames found to infer calibration from")

        if num_calib_frames == 1:
            sample_indices = [0]
        else:
            sample_indices = [
                int(round(i * (len(frame_names) - 1) / (num_calib_frames - 1)))
                for i in range(num_calib_frames)
            ]
        sampled_frames = [frame_names[i] for i in sample_indices]
        print(f"    {num_cams} cameras · sampled frames: {sampled_frames}")

        all_intrinsics: List[np.ndarray] = []
        all_extrinsics: List[np.ndarray] = []

        for frame_name in sampled_frames:
            all_images = []
            for cam_dir in camera_dirs:
                img_path = cam_dir / frame_name
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                all_images.append(str(img_path))

            with torch.no_grad():
                prediction = self.model.inference(image=all_images, use_ray_pose=True)

            if prediction.intrinsics is None or prediction.extrinsics is None:
                print(f"    ⚠ DA3 returned no poses for frame {frame_name} — skipping")
                continue

            K = np.asarray(prediction.intrinsics, dtype=np.float32)
            E = np.stack(
                [ensure_4x4_matrix(prediction.extrinsics[i]) for i in range(num_cams)]
            ).astype(np.float32)

            all_intrinsics.append(K)
            all_extrinsics.append(E)

        if not all_intrinsics:
            raise RuntimeError("DA3 failed to infer camera intrinsics/extrinsics for all sampled frames")

        K_stack = np.stack(all_intrinsics, axis=0)  # (S, C, 3, 3)
        E_stack = np.stack(all_extrinsics, axis=0)  # (S, C, 4, 4)

        intrinsics_array = np.stack(
            [average_intrinsics(K_stack[:, cam_id]) for cam_id in range(num_cams)]
        )  # (num_cams, 3, 3)
        extrinsics_array = np.stack(
            [average_extrinsics(E_stack[:, cam_id]) for cam_id in range(num_cams)]
        )  # (num_cams, 4, 4)

        print(f"    ✓ Averaged over {len(all_intrinsics)}/{num_calib_frames} frame(s)")
        print(f"    ✓ intrinsics: {intrinsics_array.shape}, extrinsics: {extrinsics_array.shape}")
        return intrinsics_array, extrinsics_array

    def align_intrinsics(
        self,
        camera_configs: List[CameraConfig],
        camera_ids: List[int],
        frame_names: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Use DA3's input_processor to obtain depth-resolution intrinsics."""
        frame_name = frame_names[0]
        all_images = [str(cfg.image_dir / frame_name) for cfg in camera_configs]
        orig_intrinsics = np.stack([cfg.intrinsics for cfg in camera_configs], axis=0)
        try:
            imgs_tensor, _, scaled_ixts = self.model.input_processor(
                all_images,
                extrinsics=None,
                intrinsics=orig_intrinsics,
            )
            _, _, H, W = imgs_tensor.shape
            depth_hw = (H, W)
            if scaled_ixts is None:
                return None, depth_hw
            return scaled_ixts.numpy().astype(np.float32), depth_hw
        except Exception as e:
            print(f"      ! DA3PoseFinder.align_intrinsics failed: {e}")
            return None, None


