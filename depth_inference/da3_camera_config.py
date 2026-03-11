from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class CameraConfig:
    """Camera calibration data."""

    intrinsics: np.ndarray  # (3, 3)
    extrinsics: np.ndarray  # (4, 4) world-to-camera
    image_dir: Path


def ensure_4x4_matrix(mat: np.ndarray) -> np.ndarray:
    """Convert a (3, 4) or (4, 4) extrinsics matrix to (4, 4)."""
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        result = np.eye(4, dtype=mat.dtype)
        result[:3, :4] = mat
        return result
    raise ValueError(f"Invalid matrix shape: {mat.shape}")


def _average_rotation_matrices(rotations: np.ndarray) -> np.ndarray:
    """Average rotation matrices via SVD projection (Chordal L2 mean)."""
    U, _, Vt = np.linalg.svd(rotations.mean(axis=0))
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def average_extrinsics(extrinsics_stack: np.ndarray) -> np.ndarray:
    """Average a stack of (N, 4, 4) world-to-camera matrices."""
    R_avg = _average_rotation_matrices(extrinsics_stack[:, :3, :3])
    t_avg = extrinsics_stack[:, :3, 3].mean(axis=0)
    result = np.eye(4, dtype=extrinsics_stack.dtype)
    result[:3, :3] = R_avg
    result[:3, 3] = t_avg
    return result


def average_intrinsics(intrinsics_stack: np.ndarray) -> np.ndarray:
    """Average a stack of (N, 3, 3) intrinsic matrices."""
    K = np.zeros((3, 3), dtype=intrinsics_stack.dtype)
    K[0, 0] = intrinsics_stack[:, 0, 0].mean()  # fx
    K[1, 1] = intrinsics_stack[:, 1, 1].mean()  # fy
    K[0, 2] = intrinsics_stack[:, 0, 2].mean()  # cx
    K[1, 2] = intrinsics_stack[:, 1, 2].mean()  # cy
    K[2, 2] = 1.0
    return K


def normalize_extrinsics_scale(
    extrinsics: np.ndarray, target_baseline: float
) -> Tuple[np.ndarray, float]:
    """Rescale w2c translations so the max pairwise camera separation equals target_baseline (m).

    Returns (scaled_extrinsics, scale_factor).
    """
    # Camera centres: p = -R^T @ t
    positions = np.stack(
        [
            -extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3]
            for i in range(len(extrinsics))
        ]
    )

    if len(positions) < 2:
        return extrinsics, 1.0

    diffs = positions[:, None] - positions[None, :]  # (N, N, 3)
    max_dist = float(np.sqrt((diffs**2).sum(axis=-1)).max())

    if max_dist < 1e-6:
        return extrinsics, 1.0

    scale = target_baseline / max_dist
    print(
        f"      [scale norm] max separation {max_dist:.4f} → scale {scale:.4f} → {target_baseline:.3f} m"
    )

    scaled = extrinsics.copy()
    scaled[:, :3, 3] *= scale
    return scaled, scale


def refine_extrinsics_icp(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_dir: Path,
    frame_idx: int = 0,
    **kwargs,
) -> np.ndarray:
    """Refine w2c extrinsics via ICP. Delegates to refine_extrinsics_icp.refine()."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import refine_extrinsics_icp as _icp

    new_c2ws = _icp.refine(
        list(extrinsics),
        list(intrinsics),
        str(Path(depth_dir).parent),
        frame=frame_idx,
        **kwargs,
    )
    return np.stack(new_c2ws).astype(np.float32)
