from pathlib import Path
from depth_inference_classes import ChunkProcessor
import torch
from da3_camera_config import CameraConfig, ensure_4x4_matrix
from typing import List, Dict
import numpy as np
import sys

import PIL.Image as PILImage
import torchvision.transforms as transforms
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "Pi3"))
from pi3.models.pi3x import Pi3X


_PIXEL_LIMIT = 255_000  # Per-frame pixel budget (same as Pi3's own default)

def _load_images_for_pi3x(
    image_paths: List[str],
    device: torch.device,
    pixel_limit: int = _PIXEL_LIMIT,
):
    """Load images, resize to the largest multiple-of-14 within *pixel_limit*.
    """
    to_tensor = transforms.ToTensor()

    # Compute uniform target size from the first image
    first = PILImage.open(image_paths[0]).convert("RGB")
    W_orig, H_orig = first.size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    k = round(W_orig * scale / 14)
    m = round(H_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_orig / H_orig:
            k -= 1
        else:
            m -= 1
    inf_W, inf_H = max(1, k) * 14, max(1, m) * 14

    tensors = []
    for path in image_paths:
        img = PILImage.open(path).convert("RGB")
        img = img.resize((inf_W, inf_H), PILImage.Resampling.LANCZOS)
        tensors.append(to_tensor(img))

    return torch.stack(tensors).to(device), inf_H, inf_W  # (N, 3, H', W')

class Pi3XChunkProcessor(ChunkProcessor):
    """ChunkProcessor implementation for Pi3X model."""
    
    def __init__(self, model: Pi3X):
        super().__init__(model)

    def process_chunk(
        self,
        camera_configs: list,  # List[CameraConfig]
        camera_ids: list,      # List[int]
        frame_names: list      # List[str]
    ) -> dict:
        """
        Run Pi3X depth inference and refine each camera's pose AND scale
        via reprojection loss to align with calibrated intrinsics/extrinsics.
        Returns depth maps aligned and scaled correctly.
        """
        num_cams = len(camera_ids)
        num_frames = len(frame_names)
        N_total = num_frames * num_cams

        # Flatten image list
        all_image_paths = [
            str(cfg.image_dir / frame_name)
            for frame_name in frame_names
            for cfg in camera_configs
        ]

        device = next(self.model.parameters()).device
        imgs, inf_H, inf_W = _load_images_for_pi3x(all_image_paths, device)

        first_img = PILImage.open(all_image_paths[0])
        orig_W, orig_H = first_img.size

        # Scale intrinsics to inference resolution
        scale_x = inf_W / orig_W
        scale_y = inf_H / orig_H
        K_scaled_list = []
        for _ in range(num_frames):
            for cfg in camera_configs:
                K = cfg.intrinsics.copy().astype(np.float32)
                K[0] *= scale_x
                K[1] *= scale_y
                K_scaled_list.append(K)
        intrinsics_t = torch.from_numpy(np.stack(K_scaled_list)).float().to(device)[None]

        # Pi3X poses c2w
        c2w_list = []
        for _ in range(num_frames):
            for cfg in camera_configs:
                w2c = ensure_4x4_matrix(cfg.extrinsics).astype(np.float32)
                c2w = np.linalg.inv(w2c)
                c2w_list.append(c2w)
        poses_t = torch.from_numpy(np.stack(c2w_list)).float().to(device)[None]

        # Masks
        mask_ray   = torch.ones(1, N_total, dtype=torch.bool, device=device)
        mask_pose  = torch.ones(1, N_total, dtype=torch.bool, device=device)
        mask_depth = torch.zeros(1, N_total, dtype=torch.bool, device=device)

        # Forward pass
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                res = self.model(
                    imgs=imgs[None],
                    intrinsics=intrinsics_t,
                    poses=poses_t,
                    mask_add_ray=mask_ray,
                    mask_add_pose=mask_pose,
                    mask_add_depth=mask_depth,
                )

        local_pts_np = res["local_points"][0].float().cpu().numpy()  # (N_total, H', W', 3)
        H, W = local_pts_np.shape[1:3]
        chunk_depths: dict = {cam_id: [] for cam_id in camera_ids}

        # --- Iterate over each camera/frame ---
        for frame_idx in range(num_frames):
            for cam_idx, cam_id in enumerate(camera_ids):
                flat_idx = frame_idx * num_cams + cam_idx
                pts_cam = local_pts_np[flat_idx]  # (H, W, 3)
                pts_flat = pts_cam.reshape(-1,3)

                # Pi3X camera → world
                c2w = c2w_list[flat_idx]
                pts_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0],1))], axis=1)
                pts_world = (c2w @ pts_h.T).T[:, :3]
                pts_world_torch = torch.tensor(pts_world, device=device, dtype=torch.float32)

                # Your camera intrinsics & extrinsics
                cfg = camera_configs[cam_idx]
                K = torch.tensor(cfg.intrinsics, device=device, dtype=torch.float32)
                w2c = torch.tensor(cfg.extrinsics, device=device, dtype=torch.float32)

                # Initialize R, t, s
                R = torch.eye(3, device=device, dtype=torch.float32, requires_grad=True)
                t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
                s = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)

                optimizer = torch.optim.Adam([R, t, s], lr=0.1)

                # --- Iterative pose + scale refinement ---
                for _ in range(50):  # iterations
                    optimizer.zero_grad()

                    # Transform points
                    pts_aligned = s * (R @ pts_world_torch.T).T + t

                    # Project into camera frame
                    pts_cam_aligned_h = w2c @ torch.cat([pts_aligned, torch.ones((pts_aligned.shape[0],1), device=device)], dim=1).T
                    pts_cam_aligned = pts_cam_aligned_h[:3].T

                    # Reprojection
                    x = pts_cam_aligned[:,0] / pts_cam_aligned[:,2]
                    y = pts_cam_aligned[:,1] / pts_cam_aligned[:,2]
                    u_proj = K[0,0]*x + K[0,2]
                    v_proj = K[1,1]*y + K[1,2]

                    # Target pixel coordinates
                    u_target = torch.linspace(0,W-1,W, device=device).repeat(H,1).reshape(-1)
                    v_target = torch.linspace(0,H-1,H, device=device).unsqueeze(1).repeat(1,W).reshape(-1)

                    # Reprojection loss
                    loss = ((u_proj - u_target)**2 + (v_proj - v_target)**2).mean()
                    loss.backward()
                    optimizer.step()

                # --- Build aligned depth map ---
                with torch.no_grad():
                    pts_aligned_final = s * (R @ pts_world_torch.T).T + t
                    pts_cam_final_h = w2c @ torch.cat([pts_aligned_final, torch.ones((pts_aligned_final.shape[0],1), device=device)], dim=1).T
                    pts_cam_final = pts_cam_final_h[:3].T
                    depth_map = pts_cam_final[:,2].reshape(H,W).cpu().numpy()
                    chunk_depths[cam_id].append(depth_map)

        return chunk_depths