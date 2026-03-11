from typing import Dict

import numpy as np


def reproject_depths_with_sim3(
    depths: Dict[int, np.ndarray],
    intrinsics: Dict[int, np.ndarray],
    extrinsics_inferred: Dict[int, np.ndarray],
    extrinsics_gt: Dict[int, np.ndarray],
    r: np.ndarray,
    t: np.ndarray,
    scale: float,
) -> Dict[int, np.ndarray]:
    """Reproject depth maps through a Sim(3) transformation.

    Backprojects each depth map to world space using inferred extrinsics,
    applies the Sim(3) transform (scale * r @ P + t), then reprojects using
    ground-truth extrinsics to produce corrected depth maps.

    Returns: Dict camera_id → corrected depth map (H, W).
    """
    reprojected = {}

    for cam_id, depth in depths.items():
        H, W = depth.shape
        depth = depth.astype(np.float32)

        K = intrinsics[cam_id]
        c2w_inferred = np.linalg.inv(extrinsics_inferred[cam_id])
        w2c_gt = extrinsics_gt[cam_id]

        # Backproject to world using inferred extrinsics
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        uv1 = np.stack([u, v, np.ones((H, W))], axis=2)  # (H, W, 3)
        ray = np.einsum("ij,hwj->hwi", np.linalg.inv(K), uv1)  # (H, W, 3)

        P_cam = np.stack(
            [
                ray[..., 0] * depth,
                ray[..., 1] * depth,
                ray[..., 2] * depth,
                np.ones((H, W)),
            ],
            axis=2,
        )  # (H, W, 4)
        P_world = np.einsum("ij,hwj->hwi", c2w_inferred, P_cam)[..., :3]  # (H, W, 3)

        # Apply Sim(3): P' = scale * (r @ P) + t
        P_transformed = np.einsum("ij,hwj->hwi", r, scale * P_world) + t  # (H, W, 3)

        # Reproject using ground-truth extrinsics
        P_gt_homo = np.concatenate([P_transformed, np.ones((H, W, 1))], axis=2)
        P_cam_gt = np.einsum("ij,hwj->hwi", w2c_gt, P_gt_homo)[..., :3]  # (H, W, 3)

        depth_corrected = P_cam_gt[..., 2]
        depth_corrected[~(np.isfinite(depth) & (depth > 0))] = 0

        reprojected[cam_id] = depth_corrected.astype(depth.dtype)

    return reprojected
