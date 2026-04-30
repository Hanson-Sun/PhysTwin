#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import warnings

import numpy as np

# Suppress torch FutureWarning about weights_only
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
except Exception:
    torch = None


def load_kpl_or_pkl(path):
    """Load a .kpl/.pkl file via torch.load first (if available), then pickle.load."""
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"Failed to load file: {path}")
        print(exc)
        sys.exit(1)


def _select_two_controllers_on_opposite_sides(ctrl0, obj0):
    """
    Select two controller points at frame 0:
    - one nearest-to-cloth on one side
    - one nearest-to-cloth on the opposite side
    """
    cloth_center = obj0.mean(axis=0)
    centered = obj0 - cloth_center

    # Use principal axis of cloth as the side split direction.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    side_axis = vh[0]
    axis_norm = np.linalg.norm(side_axis)
    if axis_norm < 1e-8:
        side_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side_axis = side_axis / axis_norm

    ctrl_proj = (ctrl0 - cloth_center) @ side_axis
    pos_candidates = np.where(ctrl_proj >= 0.0)[0]
    neg_candidates = np.where(ctrl_proj < 0.0)[0]

    dist0 = np.linalg.norm(ctrl0[:, None, :] - obj0[None, :, :], axis=-1)  # [K, N]
    ctrl_to_cloth_min = dist0.min(axis=1)  # [K]

    if len(pos_candidates) > 0 and len(neg_candidates) > 0:
        pos_idx = pos_candidates[np.argmin(ctrl_to_cloth_min[pos_candidates])]
        neg_idx = neg_candidates[np.argmin(ctrl_to_cloth_min[neg_candidates])]
        return int(pos_idx), int(neg_idx), ctrl_to_cloth_min

    # Fallback: if split fails, use two globally nearest-to-cloth controllers.
    sorted_idx = np.argsort(ctrl_to_cloth_min)
    return int(sorted_idx[0]), int(sorted_idx[1]), ctrl_to_cloth_min


def _generate_synthetic_controllers(object_points):
    """
    Generate synthetic controller points from object_points.
    Uses the two endpoints along the principal axis at each frame.
    """
    num_frames, num_objects, _ = object_points.shape
    
    # Compute center and principal axis from first frame
    obj0 = object_points[0]  # [N, 3]
    center = obj0.mean(axis=0)
    centered = obj0 - center
    
    # Find principal axis via SVD
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    principal_axis = vh[0]  # Direction of maximum variance
    
    # Create controller points by interpolating along the principal axis at each frame
    controller_points = np.zeros((num_frames, 2, 3), dtype=np.float32)
    
    for t in range(num_frames):
        obj_t = object_points[t]
        center_t = obj_t.mean(axis=0)
        centered_t = obj_t - center_t
        projections_t = centered_t @ principal_axis
        
        # Get the two extreme points
        idx_min_t = np.argmin(projections_t)
        idx_max_t = np.argmax(projections_t)
        
        controller_points[t, 0] = obj_t[idx_min_t]
        controller_points[t, 1] = obj_t[idx_max_t]
    
    return controller_points


def convert(input_path, output_path, controller_points=None, dt=0.01, neighbors_per_controller=20):
    data = load_kpl_or_pkl(input_path)

    # Handle both dict and array formats for object points
    if isinstance(data, dict):
        if "object_points" not in data:
            raise KeyError("Key 'object_points' not found in input file.")
        object_points = np.asarray(data["object_points"], dtype=np.float32)
    elif isinstance(data, np.ndarray):
        # If it's just an array, assume it's object_points
        object_points = np.asarray(data, dtype=np.float32)
    else:
        raise ValueError("Input file must contain a dict or numpy array.")

    # Use provided controller points or generate synthetic ones
    if controller_points is None:
        controller_points = _generate_synthetic_controllers(object_points)
    else:
        controller_points = np.asarray(controller_points, dtype=np.float32)

    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(
            f"object_points must have shape [T, N, 3], got {object_points.shape}."
        )
    if controller_points.ndim != 3 or controller_points.shape[-1] != 3:
        raise ValueError(
            f"controller_points must have shape [T, K, 3], got {controller_points.shape}."
        )
    if controller_points.shape[0] != object_points.shape[0]:
        raise ValueError(
            "controller_points and object_points must have the same number of frames: "
            f"{controller_points.shape[0]} vs {object_points.shape[0]}"
        )

    num_frames, num_objects, _ = object_points.shape
    if num_frames < 3:
        raise ValueError(
            f"Need at least 3 frames to compute velocity and acceleration, got {num_frames}."
        )
    if neighbors_per_controller <= 0:
        raise ValueError("neighbors_per_controller must be > 0.")

    # Forward-difference velocity: v[t] = (p[t+1] - p[t]) / dt, shape [T-1, N, 3]
    velocity = (object_points[1:] - object_points[:-1]) / dt

    # Forward-difference acceleration (force proxy): a[t] = (v[t+1] - v[t]) / dt, shape [T-2, N, 3]
    force = (velocity[1:] - velocity[:-1]) / dt

    # Align all time-dependent keys to frames where both velocity and force are valid.
    # Keep T-2 frames:
    # position -> p[1:-1], velocity -> v[1:], force -> a[:], controller -> c[1:-1]
    position_aligned = object_points[1:-1]
    velocity_aligned = velocity[1:]
    force_aligned = np.zeros_like(force, dtype=np.float32)
    ctrl_a_idx, ctrl_b_idx, ctrl_to_cloth_min = _select_two_controllers_on_opposite_sides(
        controller_points[0], object_points[0]
    )
    selected_controller_indices = [ctrl_a_idx, ctrl_b_idx]
    controller_aligned = controller_points[1:-1, selected_controller_indices, :]  # [L, 2, 3]

    nearest_k = min(neighbors_per_controller, num_objects)
    per_ctrl_distance_sum = np.zeros(2, dtype=np.float64)
    per_ctrl_distance_count = np.zeros(2, dtype=np.int64)

    for t in range(position_aligned.shape[0]):
        obj_t = position_aligned[t]  # [N, 3]
        for ci in range(2):
            ctrl_t = controller_aligned[t, ci]  # [3]
            dist = np.linalg.norm(obj_t - ctrl_t[None, :], axis=1)  # [N]

            idx_unsorted = np.argpartition(dist, kth=nearest_k - 1)[:nearest_k]
            dist_unsorted = dist[idx_unsorted]
            order = np.argsort(dist_unsorted)
            idx_sorted = idx_unsorted[order]
            dist_sorted = dist_unsorted[order]

            per_ctrl_distance_sum[ci] += dist_sorted.sum()
            per_ctrl_distance_count[ci] += dist_sorted.shape[0]
            force_aligned[t, idx_sorted] = force[t, idx_sorted]

    # 1. Flip the Z coordinate in position_aligned
    position_aligned[:, :, 2] = -position_aligned[:, :, 2]

    # 2. Create the box with x and y range
    max_abs_x = np.max(np.abs(position_aligned[:, :, 0]))
    max_abs_y = np.max(np.abs(position_aligned[:, :, 1]))
    a = max(max_abs_x, max_abs_y) + 0.03

    x_vals = np.linspace(-a, a, 100)  # Create a grid for X
    y_vals = np.linspace(-a, a, 100)  # Create a grid for Y
    X, Y = np.meshgrid(x_vals, y_vals)  # Create 2D mesh grid
    box_points = np.vstack([X.ravel(), Y.ravel()]).T  # Stack into shape (10000, 2)
    box_points = np.hstack([box_points, np.zeros((box_points.shape[0], 1))])  # Add Z=0

    # Box feats (same shape as box_points)
    box_feats = np.zeros((box_points.shape[0], 3), dtype=np.float32)
    box_feats[:, 2] = 1  # Set the third column to 1

    # 3. Save the results
    feats = np.ones((num_objects, 1), dtype=np.float32)

    np.savez_compressed(
        output_path,
        position=position_aligned,
        velocity=velocity_aligned,
        feats=feats,
        force=force_aligned,
        box_points=box_points,
        box_feats=box_feats,
    )

    print(f"Saved: {output_path}")
    print(f"position shape: {position_aligned.shape}")
    print(f"velocity shape: {velocity_aligned.shape}")
    print(f"force shape: {force_aligned.shape}")
    print(f"feats shape: {feats.shape}")
    print(f"box_points shape: {box_points.shape}")
    print(f"box_feats shape: {box_feats.shape}")
    print(
        "selected controllers (from frame 0): "
        f"{selected_controller_indices[0]}, {selected_controller_indices[1]}"
    )
    print(
        "controller-to-cloth distance at frame 0: "
        f"{ctrl_to_cloth_min[selected_controller_indices[0]]:.6f}, "
        f"{ctrl_to_cloth_min[selected_controller_indices[1]]:.6f}"
    )
    print(
        f"mean distance of {nearest_k} selected object points to controller "
        f"{selected_controller_indices[0]}: "
        f"{(per_ctrl_distance_sum[0] / max(per_ctrl_distance_count[0], 1)):.6f}"
    )
    print(
        f"mean distance of {nearest_k} selected object points to controller "
        f"{selected_controller_indices[1]}: "
        f"{(per_ctrl_distance_sum[1] / max(per_ctrl_distance_count[1], 1)):.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert inference.pkl to npz with position/velocity/feats/force."
    )
    parser.add_argument("--input", required=True, help="Path to input .pkl/.kpl file (inference.pkl)")
    parser.add_argument("--output", default=None, help="Path to output .npz file")
    parser.add_argument("--controller-data", default=None, help="Path to controller data file (final_data.pkl)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for finite differences")
    parser.add_argument(
        "--neighbors_per_controller",
        type=int,
        default=20,
        help="How many nearest object points to keep per selected controller per frame.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)

    if args.output is None:
        root, _ = os.path.splitext(input_path)
        output_path = root + ".npz"
    else:
        output_path = args.output

    # Load controller points if provided
    controller_points = None
    if args.controller_data is not None:
        if os.path.exists(args.controller_data):
            try:
                data = load_kpl_or_pkl(args.controller_data)
                if isinstance(data, dict) and "controller_points" in data:
                    controller_points = data["controller_points"]
                    print(f"Loaded controller_points from: {args.controller_data}")
                    print(f"Controller points shape: {np.asarray(controller_points).shape}")
                else:
                    print(f"Warning: {args.controller_data} does not contain 'controller_points' key")
            except Exception as e:
                print(f"Warning: Failed to load controller data: {e}")
        else:
            print(f"Warning: Controller data file not found: {args.controller_data}")

    convert(
        input_path=input_path,
        output_path=output_path,
        controller_points=controller_points,
        dt=args.dt,
        neighbors_per_controller=args.neighbors_per_controller,
    )


if __name__ == "__main__":
    main()
