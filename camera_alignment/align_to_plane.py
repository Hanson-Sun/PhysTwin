"""
Align camera extrinsics so the detected floor plane becomes z=0.
This transforms the coordinate system so the physics pipeline doesn't need custom collision logic.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from argparse import ArgumentParser
import pickle
import json
import os
import shutil


def compute_plane_alignment_transform(plane_eq: np.ndarray) -> np.ndarray:
    """
    Compute a 4x4 transformation matrix that aligns the given plane with z=0.
    
    The transform moves the plane to z=0 and rotates so the normal points along +Z.
    
    Args:
        plane_eq: [a, b, c, d] for plane equation ax + by + cz + d = 0
        
    Returns:
        T: 4x4 transformation matrix such that T @ plane becomes [0, 0, 1, 0] (z=0)
    """
    a, b, c, d = plane_eq
    
    # Normal vector (normalize)
    normal = np.array([a, b, c])
    normal_len = np.linalg.norm(normal)
    normal = normal / normal_len
    
    # Point on the plane: find one solution to ax + by + cz + d = 0
    # Try z=0 first: ax + by + d = 0
    if abs(a) > abs(b) and abs(a) > 1e-6:
        plane_point = np.array([-d / a, 0.0, 0.0])
    elif abs(b) > 1e-6:
        plane_point = np.array([0.0, -d / b, 0.0])
    else:  # c is largest
        plane_point = np.array([0.0, 0.0, -d / c if abs(c) > 1e-6 else 0.0])
    
    # Compute rotation: align plane normal with [0, 0, 1]
    target_normal = np.array([0.0, 0.0, 1.0])
    
    # Check if already aligned
    axis = np.cross(normal, target_normal)
    axis_len = np.linalg.norm(axis)
    
    if axis_len < 1e-6:
        # Normal is parallel to target (up or down)
        if np.dot(normal, target_normal) > 0:
            R = np.eye(3)  # Already aligned
        else:
            # Flipped: rotate 180 degrees
            R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        # Rodrigues rotation formula
        axis = axis / axis_len
        angle = np.arccos(np.clip(np.dot(normal, target_normal), -1, 1))
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # Combine: translate plane point to origin, then rotate
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ plane_point
    
    return T


def apply_transform_to_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation to points of various shapes."""
    original_shape = points.shape
    
    if points.ndim == 3:  # (T, N, 3)
        points_flat = points.reshape(-1, 3)
    elif points.ndim == 2:  # (N, 3)
        points_flat = points
    else:
        raise ValueError(f"Unexpected points shape: {original_shape}")
    
    # Transform
    pts_homo = np.concatenate([points_flat, np.ones((points_flat.shape[0], 1))], axis=1)
    pts_transformed_homo = (T @ pts_homo.T).T
    pts_transformed = pts_transformed_homo[:, :3]
    
    # Reshape back
    if points.ndim == 3:
        pts_transformed = pts_transformed.reshape(original_shape)
    
    return pts_transformed


def align_to_plane(base_path: str, case_name: str, planes_json: Optional[str] = None) -> Tuple[bool, Dict]:
    """
    Load plane detection, compute alignment transform, and apply to all data.
    
    Args:
        base_path: Path to data directory (e.g., ./data/different_types)
        case_name: Case name
        planes_json: Path to environment_planes.json (auto-detected if None)
        
    Returns:
        success: Whether alignment was successful
        info: Dictionary with alignment details
    """
    case_path = f"{base_path}/{case_name}"
    pkl_path = f"{case_path}/final_data.pkl"
    calib_path = f"{case_path}/calibrate.pkl"
    alignment_marker = f"{case_path}/.aligned_to_plane"
    
    info = {}
    
    # Check if already aligned
    if os.path.exists(alignment_marker):
        info["status"] = "Already aligned (marker file exists)"
        print(f"⚠ Case '{case_name}' has already been aligned to plane. Skipping.")
        return True, info
    
    # Find planes JSON
    if planes_json is None:
        planes_json = f"{case_path}/environment_planes.json"
    
    # Load plane
    if not os.path.exists(planes_json):
        info["status"] = f"Plane file not found, skipping alignment (will use default z=0 plane)"
        print(f"⚠ {info['status']}")
        return True, info
    
    with open(planes_json) as f:
        planes_data = json.load(f)
    
    if not planes_data.get('planes'):
        info["status"] = "No planes detected, skipping alignment (will use default z=0 plane)"
        print(f"⚠ {info['status']}")
        return True, info
    
    plane_eq = np.array(planes_data['planes'][0]['equation'])
    print(f"Detected plane equation: {plane_eq}")
    
    # Compute alignment
    T = compute_plane_alignment_transform(plane_eq)
    print(f"\nAlignment transform (floor → z=0):")
    print(T)
    
    # Load and transform calibration
    with open(calib_path, "rb") as f:
        c2ws = pickle.load(f)
    
    c2ws = [np.array(c2w) for c2w in c2ws]
    c2ws_aligned = []
    
    for i, c2w in enumerate(c2ws):
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        c2w_aligned = T @ c2w
        c2ws_aligned.append(c2w_aligned)
        
        # Print camera position before/after
        pos_before = c2w[:3, 3]
        pos_after = c2w_aligned[:3, 3]
        print(f"  Camera {i}: position {pos_before} → {pos_after}")
    
    # Load and transform object data
    with open(pkl_path, "rb") as f:
        track_data = pickle.load(f)
    
    point_keys = ["object_points", "controller_points", "surface_points", "interior_points"]
    
    for key in point_keys:
        if key in track_data and track_data[key] is not None:
            pts_before = track_data[key]
            if len(pts_before) > 0 and pts_before.shape[-1] >= 3:
                pts_after = apply_transform_to_points(pts_before, T)
                track_data[key] = pts_after
                
                # Log transformation
                z_before = (pts_before[..., 2].min(), pts_before[..., 2].max())
                z_after = (pts_after[..., 2].min(), pts_after[..., 2].max())
                print(f"  {key}: Z range {z_before} → {z_after}")
    
    # Save aligned data back
    # First, create backups of original files
    calib_backup = f"{calib_path}.unaligned"
    pkl_backup = f"{pkl_path}.unaligned"
    
    if os.path.exists(calib_path) and not os.path.exists(calib_backup):
        shutil.copy2(calib_path, calib_backup)
        print(f"Backed up original calibration to {calib_backup}")
    
    if os.path.exists(pkl_path) and not os.path.exists(pkl_backup):
        shutil.copy2(pkl_path, pkl_backup)
        print(f"Backed up original object data to {pkl_backup}")
    
    # Now save aligned data
    with open(calib_path, "wb") as f:
        pickle.dump(c2ws_aligned, f)
    print(f"Saved aligned calibration to {calib_path}")
    
    with open(pkl_path, "wb") as f:
        pickle.dump(track_data, f)
    print(f"Saved aligned object data to {pkl_path}")
    
    # Write marker file to prevent re-alignment
    with open(alignment_marker, "w") as f:
        f.write(f"Aligned to plane on {case_name}\n")
        f.write(f"Plane equation: {plane_eq.tolist()}\n")
    print(f"Created alignment marker: {alignment_marker}")
    
    info["status"] = "success"
    info["plane_equation"] = plane_eq.tolist()
    return True, info


if __name__ == "__main__":
    parser = ArgumentParser(description="Align extrinsics to detected plane (floor → z=0)")
    parser.add_argument("--base_path", type=str, default="./data/different_types",
                       help="Base data path")
    parser.add_argument("case_name", type=str,
                       help="Case name (e.g., single_lift_rope_new)")
    parser.add_argument("--planes_json", type=str, default=None,
                       help="Path to environment_planes.json (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    success, info = align_to_plane(args.base_path, args.case_name, args.planes_json)
    
    if success:
        print("\n✓ Successfully aligned extrinsics to plane!")
    else:
        print(f"\n✗ Failed: {info.get('status', 'Unknown error')}")
        exit(1)
