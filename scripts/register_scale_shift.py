#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import numpy as np

def load_trajectory(path: Path, key: str | None):
    """Loads trajectory from PKL, NPZ, or NPY."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            # Common keys in your pipeline
            for k in [key, "object_points", "position", "arr_0"]:
                if k in data:
                    arr = data[k]
                    break
            else:
                raise KeyError(f"No usable key found in {path}")
    else: # Default to PKL
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            for k in [key, "object_points", "position"]:
                if k in data:
                    arr = data[k]
                    break
            else:
                raise KeyError(f"No usable key in PKL. Available: {list(data.keys())}")
        else:
            arr = data

    arr = np.asarray(arr, dtype=np.float32)
    # Ensure [T, N, 3]
    if arr.ndim == 2: # Handle case where it might be (N, 3) for a single frame
        arr = arr[None, ...]
    return arr

def register_scale_shift(source: np.ndarray, reference: np.ndarray):
    t = min(source.shape[0], reference.shape[0])
    n = min(source.shape[1], reference.shape[1])

    src_crop = source[:t, :n]
    ref_crop = reference[:t, :n]

    # --- Scale (unchanged) ---
    src_min = np.percentile(src_crop, 1, axis=(0, 1))
    src_max = np.percentile(src_crop, 99, axis=(0, 1))
    ref_min = np.percentile(ref_crop, 1, axis=(0, 1))
    ref_max = np.percentile(ref_crop, 99, axis=(0, 1))

    src_main_dim = np.max(src_max - src_min)
    ref_main_dim = np.max(ref_max - ref_min)
    scale = float(ref_main_dim / (src_main_dim + 1e-12))

    # --- Shift: pick best center estimator independently per axis ---
    src_flat = src_crop.reshape(-1, 3)
    ref_flat = ref_crop.reshape(-1, 3)

    axis_names = ["X", "Y", "Z"]

    center_estimators = {
        "median": (
            np.median(src_flat, axis=0),
            np.median(ref_flat, axis=0),
        ),
        # "p10/p90": (
        #     (np.percentile(src_flat, 10, axis=0) + np.percentile(src_flat, 90, axis=0)) / 2.0,
        #     (np.percentile(ref_flat, 10, axis=0) + np.percentile(ref_flat, 90, axis=0)) / 2.0,
        # ),
        # "p20/p80": (
        #     (np.percentile(src_flat, 20, axis=0) + np.percentile(src_flat, 80, axis=0)) / 2.0,
        #     (np.percentile(ref_flat, 20, axis=0) + np.percentile(ref_flat, 80, axis=0)) / 2.0,
        # ),
        "p1/p99": (
            (src_min + src_max) / 2.0,
            (ref_min + ref_max) / 2.0,
        ),
    }

    best_shift = np.zeros(3, dtype=np.float64)
    best_names = []

    for axis in range(3):
        best_name, best_rmse = None, np.inf
        for name, (src_center, ref_center) in center_estimators.items():
            shift_axis = ref_center[axis] - scale * src_center[axis]
            # Build a full shift with this axis swapped in, others zeroed for isolated eval
            trial_shift = best_shift.copy()
            trial_shift[axis] = shift_axis
            aligned = (source * scale) + trial_shift[None, None, :]
            rmse = np.sqrt(np.mean((aligned[:t, :n, axis] - ref_crop[..., axis]) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_shift[axis] = shift_axis
        best_names.append(f"{axis_names[axis]}={best_name}({best_rmse:.4f})")

    print(f"  Center estimators: {', '.join(best_names)}")

    best_shift[0] += 0.05

    aligned = (source * scale) + best_shift[None, None, :]
    rmse_before = np.sqrt(np.mean((source[:t, :n] - ref_crop) ** 2))
    rmse_after = np.sqrt(np.mean((aligned[:t, :n] - ref_crop) ** 2))

    return aligned.astype(np.float32), scale, best_shift.astype(np.float32), float(rmse_before), float(rmse_after)

def main():
    parser = argparse.ArgumentParser(description="Register simulation to reference data.")
    parser.add_argument("--source-pkl", required=True)
    parser.add_argument("--reference", required=True, help="Path to final_data.pkl or .npy")
    parser.add_argument("--reverse-z", action="store_true", help="Flip Z axis of source before registration")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    src_path = Path(args.source_pkl)
    ref_path = Path(args.reference)

    source = load_trajectory(src_path, None)
    reference = load_trajectory(ref_path, "object_points")

    if args.reverse_z:
        print("[Info] Reversing Z-axis for source trajectory.")
        source[..., 2] *= -1

    aligned, scale, shift, r_before, r_after = register_scale_shift(source, reference)

    output_path = Path(args.output) if args.output else src_path
    with open(output_path, "wb") as f:
        pickle.dump(aligned, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Registration Report:")
    print(f"  Scale: {scale:.6f}")
    print(f"  Shift: {shift}")
    print(f"  RMSE:  {r_before:.6f} -> {r_after:.6f}")
    
    if r_after > 0.1:
        print("Warning: RMSE is high. This usually means axes (X,Y,Z) are swapped or inverted.")

if __name__ == "__main__":
    main()