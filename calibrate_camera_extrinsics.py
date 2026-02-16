"""
Calibrate camera extrinsics to ensure depth projections don't go above z=0.
Run early in pipeline (after depth inference) to adjust camera position.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from argparse import ArgumentParser
import pickle
import os


def calculate_z_shift(points: np.ndarray, method: str = "inlier", 
                     std_threshold: float = 3.0) -> float:
    """Calculate positive z-shift to ensure all points have positive z values."""
    z = points[..., 2]
    
    if method == "inlier":
        mean, std = np.mean(z), np.std(z)
        z_inliers = z[np.abs(z - mean) <= std_threshold * std]
        min_z = np.min(z_inliers) if len(z_inliers) > 0 else np.min(z)
    elif method == "max":
        min_z = np.min(z)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max(0.0, -min_z + 0.01) if min_z <= 0 else 0.0


def _shift_points_z(points: np.ndarray, amount: float) -> np.ndarray:
    """Shift points along z-axis."""
    shifted = points.copy()
    shifted[..., 2] += amount
    return shifted


def adjust_camera_extrinsics(extrinsics: np.ndarray, shift_amount: float) -> np.ndarray:
    """Shift camera position along world z-axis."""
    ext = extrinsics.copy()
    ext[2, 3] += shift_amount
    return ext


def calibrate_track_data(track_data: Dict, shift_amount: Optional[float] = None, 
                        method: str = "inlier", std_threshold: float = 3.0) -> Tuple[Dict, float, Dict]:
    """Calculate shift needed to ensure all points have positive z, optionally apply it."""
    point_keys = ["object_points", "surface_points", "interior_points", "controller_points"]
    
    def extract_points(key):
        pts = track_data.get(key)
        if isinstance(pts, np.ndarray) and pts.size > 0 and pts.shape[-1] >= 3:
            return pts.reshape(-1, 3)
        return None
    
    point_list = [p for p in (extract_points(k) for k in point_keys) if p is not None]
    if not point_list:
        return track_data, 0.0, {"shifted": False, "reason": "No points found"}
    
    all_points = np.concatenate(point_list, axis=0)
    z_before = (all_points[..., 2].min(), all_points[..., 2].max())
    
    # Check if already calibrated (all z > 0)
    if z_before[0] > 0:
        return track_data, 0.0, {"shifted": False, "reason": "Already calibrated (all z > 0)"}
    
    if shift_amount is None:
        shift_amount = calculate_z_shift(all_points, method=method, std_threshold=std_threshold)
    
    info = {"shifted": shift_amount > 0, "shift_amount": shift_amount, "z_before": z_before}
    
    if shift_amount > 0:
        for key in point_keys:
            pts = extract_points(key)
            if pts is not None:
                track_data[key] = _shift_points_z(track_data[key], shift_amount)
        all_points[..., 2] -= shift_amount
        info["z_after"] = (all_points[..., 2].min(), all_points[..., 2].max())
    
    return track_data, shift_amount, info


def calibrate_mesh(mesh_path: str, shift_amount: float) -> bool:
    """Shift mesh vertices along z-axis. Handles GLB (meshes) and PLY (point clouds)."""
    import trimesh
    from plyfile import PlyData
    
    if not os.path.exists(mesh_path):
        print(f"Warning: Mesh file not found at {mesh_path}")
        return False
    
    try:
        if mesh_path.endswith(".ply"):
            ply = PlyData.read(mesh_path)
            ply['vertex']['z'] -= shift_amount
            ply.write(mesh_path)
            print(f"  ✓ Shifted PLY vertices by {shift_amount:.6f} (Z-axis)")
            return True
        
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            for geom in mesh.geometry.values():
                if hasattr(geom, 'vertices'):
                    geom.vertices[:, 2] -= shift_amount
        else:
            mesh.vertices[:, 2] -= shift_amount
        
        mesh.export(mesh_path)
        print(f"  ✓ Shifted mesh vertices by {shift_amount:.6f} (Z-axis)")
        return True
    except Exception as e:
        print(f"  Error calibrating mesh: {e}")
        return False


if __name__ == "__main__":
    parser = ArgumentParser(description="Calibrate camera extrinsics for depth projection")
    parser.add_argument("--base_path", type=str, required=True, 
                       help="Base data path (e.g., ../data or /path/to/data)")
    parser.add_argument("--case_name", type=str, required=True,
                       help="Case name (e.g., cloth_double_bend)")
    parser.add_argument("--method", choices=["inlier", "max"], default="inlier",
                       help="Method for shift calculation")
    parser.add_argument("--std_threshold", type=float, default=3.0)
    parser.add_argument("--shift_amount", type=float, default=None,
                       help="Override calculated shift (for manual override)")
    args = parser.parse_args()
    
    case_path = f"{args.base_path}/{args.case_name}"
    pkl_path = f"{case_path}/final_data.pkl"
    calib_path = f"{case_path}/calibrate.pkl"
    
    with open(pkl_path, "rb") as f:
        track_data = pickle.load(f)
    print(f"Loaded point data from {pkl_path}")
    
    with open(calib_path, "rb") as f:
        c2ws = pickle.load(f)
    
    if not isinstance(c2ws, list):
        raise TypeError(f"calibrate.pkl should be a list, got {type(c2ws)}")
    
    c2ws = [np.array(c2w) for c2w in c2ws]
    print(f"Loaded {len(c2ws)} c2w matrices from {calib_path}")
    
    track_data, shift, info = calibrate_track_data(
        track_data, shift_amount=args.shift_amount, method=args.method, 
        std_threshold=args.std_threshold
    )
    
    print(f"\n{'='*50}\nCamera Extrinsics Calibration\n{'='*50}")
    print(f"Case: {args.case_name}")
    if "z_before" in info:
        print(f"Z range BEFORE: {info['z_before'][0]:.6f} to {info['z_before'][1]:.6f}")
    print(f"Shift amount: {shift:.6f}")
    if "z_after" in info:
        print(f"Z range AFTER: {info['z_after'][0]:.6f} to {info['z_after'][1]:.6f}")
    
    if not info.get("shifted"):
        print(f"\n⚠ Skipped: {info.get('reason', 'No shift needed')}")
    else:
        c2ws = [adjust_camera_extrinsics(c2w, shift) for c2w in c2ws]
        
        with open(calib_path, "wb") as f:
            pickle.dump(c2ws, f)
        with open(pkl_path, "wb") as f:
            pickle.dump(track_data, f)
        
        print(f"\n✓ Shifted camera and points by {shift:.6f}")
        print(f"✓ Saved {calib_path}\n✓ Saved {pkl_path}")
        
        for mesh_path in [f"{case_path}/shape/object.glb", f"{case_path}/shape/matching/final_mesh.glb"]:
            if os.path.exists(mesh_path) and calibrate_mesh(mesh_path, shift):
                print(f"✓ Saved {mesh_path}")
            elif not os.path.exists(mesh_path):
                print(f"  (Mesh not found: {mesh_path})")
    else:
        print(f"\nNo shift needed")
