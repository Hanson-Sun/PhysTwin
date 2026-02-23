import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

@dataclass
class CameraConfig:
    """Camera calibration data."""
    intrinsics: np.ndarray  # (3, 3)
    extrinsics: np.ndarray  # (4, 4) world-to-camera
    image_dir: Path
    
def ensure_4x4_matrix(mat: np.ndarray) -> np.ndarray:
    """Convert (3, 4) extrinsics to (4, 4) homogeneous matrix."""
    if mat.shape == (4, 4):
        return mat
    elif mat.shape == (3, 4):
        result = np.eye(4, dtype=mat.dtype)
        result[:3, :4] = mat
        return result
    else:
        raise ValueError(f"Invalid matrix shape: {mat.shape}")

def average_rotation_matrices(rotations: np.ndarray) -> np.ndarray:
    """Average a set of rotation matrices via SVD projection (Chordal L2 mean).
    """
    R_mean = rotations.mean(axis=0)  # (3, 3) element-wise mean
    U, _, Vt = np.linalg.svd(R_mean)
    R_avg = U @ Vt
    # Ensure proper rotation (det = +1), not a reflection
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def average_extrinsics(extrinsics_stack: np.ndarray) -> np.ndarray:
    """Average a set of 4x4 world-to-camera extrinsic matrices.
    """
    rotations = extrinsics_stack[:, :3, :3]   # (N, 3, 3)
    translations = extrinsics_stack[:, :3, 3]  # (N, 3)

    R_avg = average_rotation_matrices(rotations)
    t_avg = translations.mean(axis=0)

    result = np.eye(4, dtype=extrinsics_stack.dtype)
    result[:3, :3] = R_avg
    result[:3, 3] = t_avg
    return result


def average_intrinsics(intrinsics_stack: np.ndarray) -> np.ndarray:
    """Average a set of 3x3 camera intrinsic matrices.
    """
    fx = float(intrinsics_stack[:, 0, 0].mean())
    fy = float(intrinsics_stack[:, 1, 1].mean())
    cx = float(intrinsics_stack[:, 0, 2].mean())
    cy = float(intrinsics_stack[:, 1, 2].mean())

    K = np.zeros((3, 3), dtype=intrinsics_stack.dtype)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K
    
def infer_calibration(
    model: DepthAnything3,
    case_dir: Path,
    frame_names: List[str],
    num_calib_frames: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer intrinsics and extrinsics by averaging over multiple frames.
    """
    num_calib_frames = max(1, min(num_calib_frames, len(frame_names)))
    print(f"\n[*] Inferring camera calibration by averaging over {num_calib_frames} frame(s)...")

    color_dir = case_dir / "color"
    camera_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    num_cams = len(camera_dirs)

    if not frame_names:
        raise ValueError("No frames found to infer calibration from")

    # Pick evenly-spaced indices (always includes first and last when > 1)
    if num_calib_frames == 1:
        sample_indices = [0]
    else:
        sample_indices = [
            int(round(i * (len(frame_names) - 1) / (num_calib_frames - 1)))
            for i in range(num_calib_frames)
        ]
    sampled_frames = [frame_names[i] for i in sample_indices]
    print(f"    {num_cams} cameras · sampled frames: {sampled_frames}")

    # Accumulate per-frame results: list of (num_cams, 3, 3) and (num_cams, 4, 4)
    all_intrinsics: List[np.ndarray] = []   # each entry: (num_cams, 3, 3)
    all_extrinsics: List[np.ndarray] = []   # each entry: (num_cams, 4, 4)
    depth_hw: Optional[Tuple[int, int]] = None

    for frame_name in sampled_frames:
        all_images = []
        for cam_dir in camera_dirs:
            img_path = cam_dir / frame_name
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            all_images.append(str(img_path))

        with torch.no_grad():
            prediction = model.inference(image=all_images, use_ray_pose=True)

        if prediction.intrinsics is None or prediction.extrinsics is None:
            print(f"    ⚠ DA3 returned no poses for frame {frame_name} — skipping")
            continue

        if depth_hw is None and prediction.depth is not None:
            depth_hw = prediction.depth[0].shape  # (H, W)

        K = np.asarray(prediction.intrinsics, dtype=np.float32) 
        E = np.stack(
            [ensure_4x4_matrix(prediction.extrinsics[i]) for i in range(num_cams)]
        ).astype(np.float32)                                

        all_intrinsics.append(K)
        all_extrinsics.append(E)

    if not all_intrinsics:
        raise RuntimeError("DA3 failed to infer camera intrinsics/extrinsics for all sampled frames")

    # Shape after stack: (num_samples, num_cams, 3, 3) / (num_samples, num_cams, 4, 4)
    K_stack = np.stack(all_intrinsics, axis=0)  # (S, C, 3, 3)
    E_stack = np.stack(all_extrinsics, axis=0)  # (S, C, 4, 4)

    intrinsics_array = np.stack(
        [average_intrinsics(K_stack[:, cam_id]) for cam_id in range(num_cams)]
    )  # (num_cams, 3, 3)

    extrinsics_array = np.stack(
        [average_extrinsics(E_stack[:, cam_id]) for cam_id in range(num_cams)]
    )  # (num_cams, 4, 4)

    n_used = len(all_intrinsics)
    print(f"    ✓ Averaged over {n_used}/{num_calib_frames} frame(s)")
    print(f"    ✓ intrinsics: {intrinsics_array.shape}, extrinsics: {extrinsics_array.shape}")
    if depth_hw:
        print(f"    ✓ depth-map resolution: {depth_hw[1]}x{depth_hw[0]}")
    return intrinsics_array, extrinsics_array
    
def align_camera_intrinsics(
    model: DepthAnything3,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """
    Compute intrinsics and depth-map dimensions at DA3's processed resolution.

    Returns:
        (scaled_intrinsics, depth_hw) or (None, None) on failure.
    """
    frame_name = frame_names[0]
    all_images = [str(cfg.image_dir / frame_name) for cfg in camera_configs]
    orig_intrinsics = [cfg.intrinsics for cfg in camera_configs]

    try:
        imgs_tensor, _, scaled_ixts = model.input_processor(
            all_images,
            extrinsics=None,
            intrinsics=np.stack(orig_intrinsics, axis=0),
        )
        # imgs_tensor: (N, 3, H, W)
        _, _, H, W = imgs_tensor.shape
        depth_hw: Tuple[int, int] = (H, W)

        if scaled_ixts is None:
            return None, depth_hw

        scaled_intrinsics = scaled_ixts.numpy().astype(np.float32)  # (N, 3, 3)
        return scaled_intrinsics, depth_hw

    except Exception as e:
        print(f"      ! align_camera_intrinsics failed: {e}")
        return None, None


# ── DUSt3R calibration wrapper ────────────────────────────────────────────────

def infer_calibration_dust3r(
    case_dir: Path,
    frame_names: List[str],
    num_calib_frames: int = 5,
    frame: int = 0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer camera intrinsics and extrinsics using DUSt3R instead of DA3.
    Delegates to calibrate_with_dust3r.calibrate().
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    _sys.path.insert(0, str(Path(__file__).parent.parent / "dust3r"))
    from calibrate_with_dust3r import calibrate
    from dust3r.model import AsymmetricCroCo3DStereo

    model = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    Ks, c2ws = calibrate(model, str(case_dir), frame=frame, **kwargs)
    intrinsics_array = np.stack(Ks).astype(np.float32)      # (N, 3, 3)
    c2w_array = np.stack(c2ws).astype(np.float32)           # (N, 4, 4) c2w
    # DUSt3R returns cam-to-world; CameraConfig.extrinsics (and infer_calibration)
    # expect world-to-camera, so invert here.
    extrinsics_array = np.linalg.inv(c2w_array)             # (N, 4, 4) w2c
    return intrinsics_array, extrinsics_array



def refine_extrinsics_icp(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_dir: Path,
    frame_idx: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Refine camera extrinsics (cam-to-world) via ICP alignment of background depth.
    Delegates to refine_extrinsics_icp.refine().
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    import refine_extrinsics_icp as _icp

    case_dir = str(Path(depth_dir).parent)
    new_c2ws = _icp.refine(
        list(extrinsics), list(intrinsics), case_dir, frame=frame_idx, **kwargs
    )
    return np.stack(new_c2ws).astype(np.float32)


def normalize_extrinsics_scale(extrinsics: np.ndarray, target_baseline: float) -> Tuple[np.ndarray, float]:
    """
    Rescale extrinsic translations so the maximum pairwise camera separation
    equals ``target_baseline`` (metres).

    Works on world-to-camera (w2c) matrices.  The camera position in world
    space is  p = -R^T @ t  (i.e. the last column of the corresponding c2w).
    Scaling all positions by ``s`` is equivalent to multiplying every w2c
    translation vector by ``s``, which is what we do here.

    Returns the rescaled extrinsics array and the scale factor applied.
    """
    # Extract camera-centre positions from w2c matrices
    positions = np.stack([-extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3]
                          for i in range(len(extrinsics))])  # (N, 3)

    if len(positions) < 2:
        print("      [scale norm] Only one camera — skipping scale normalisation")
        return extrinsics, 1.0

    # Maximum pairwise distance
    max_dist = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d > max_dist:
                max_dist = d

    if max_dist < 1e-6:
        print("      [scale norm] Camera positions are degenerate — skipping scale normalisation")
        return extrinsics, 1.0

    scale = target_baseline / max_dist
    print(f"      [scale norm] Max camera separation: {max_dist:.4f} units → "
          f"rescaling by {scale:.4f} to reach {target_baseline:.3f} m")

    scaled = extrinsics.copy()
    scaled[:, :3, 3] *= scale   # multiply w2c translation vectors
    return scaled, scale
