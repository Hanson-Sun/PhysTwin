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


def _load_npz(path):
    try:
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    except Exception as exc:
        print(f"Failed to load NPZ file: {path}")
        print(exc)
        sys.exit(1)


def _extract_controller_points(data, key):
    if isinstance(data, dict):
        if key in data:
            return np.asarray(data[key], dtype=np.float32)
        if key != "controller_points" and "controller_points" in data:
            return np.asarray(data["controller_points"], dtype=np.float32)
        raise KeyError(
            f"Controller key '{key}' was not found in controller data dict."
        )

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr

    raise ValueError(
        "Controller data must be a dict containing controller points, or an array with shape [T, K, 3]."
    )


def _reconstruct_endpoints(position, velocity, dt, mode):
    """
    Reconstruct missing boundary frames around aligned position.

    mode:
      - none: keep as-is
      - prepend: estimate first frame as p0 ~= p1 - dt * v1
      - append: append last frame as p_last = p_end + dt * v_end
      - both: apply prepend and append
    """
    if mode == "none":
        return position

    if velocity is None:
        raise ValueError(
            "Endpoint reconstruction requested but velocity data is unavailable."
        )

    if velocity.ndim != 3 or velocity.shape[-1] != 3:
        raise ValueError(f"velocity must have shape [L, N, 3], got {velocity.shape}.")

    if velocity.shape[1:] != position.shape[1:]:
        raise ValueError(
            "velocity and position point dimensions must match: "
            f"{velocity.shape[1:]} vs {position.shape[1:]}"
        )

    if velocity.shape[0] < 1:
        raise ValueError("velocity must contain at least one frame for reconstruction.")

    parts = []
    if mode in ("prepend", "both"):
        first = position[0] - dt * velocity[0]
        parts.append(first[None])

    parts.append(position)

    if mode in ("append", "both"):
        last = position[-1] + dt * velocity[-1]
        parts.append(last[None])

    return np.concatenate(parts, axis=0)


def convert(
    input_path,
    output_path,
    position_key="position",
    velocity_key="velocity",
    unflip_z=True,
    output_format="dict",
    dict_key="object_points",
    reconstruct_endpoints="none",
    dt=0.01,
    controller_points=None,
    controller_key="controller_points",
    max_frames=None,
):
    npz_data = _load_npz(input_path)

    print(npz_data[position_key].shape)

    if position_key not in npz_data:
        available_keys = ", ".join(sorted(npz_data.keys()))
        raise KeyError(
            f"Key '{position_key}' not found in NPZ. Available keys: {available_keys}"
        )

    position = np.asarray(npz_data[position_key], dtype=np.float32).copy()
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError(
            f"position must have shape [T, N, 3], got {position.shape}."
        )

    if unflip_z:
        position[:, :, 2] = -position[:, :, 2]

    velocity = None
    if velocity_key in npz_data:
        velocity = np.asarray(npz_data[velocity_key], dtype=np.float32)

    object_points = _reconstruct_endpoints(position, velocity, dt, reconstruct_endpoints)

    # Truncate controller_points to match object_points frame count if it is longer.
    # if max_frames is not None:
    #     print("TRUNCATINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
    #     obj_t = object_points.shape[0]

    #     def repeat_to_length(arr, target_len):
    #         cur = arr.shape[0]
    #         reps = int(np.ceil(target_len / cur))
    #         tiled = np.concatenate([arr] * reps, axis=0)
    #         return tiled[:target_len]


    #     if obj_t < max_frames:
    #         print(
    #             f"[Info] object_points has {obj_t} frames, repeating to {max_frames}."
    #         )
    #         object_points = repeat_to_length(object_points, max_frames)

    print(f"Final object_points shape: {object_points.shape}")

    if output_format == "dict":
        payload = {dict_key: object_points}
        if controller_points is not None:
            payload[controller_key] = np.asarray(controller_points, dtype=np.float32)
    elif output_format == "array":
        payload = object_points
        print(object_points.shape)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {output_path}")
    print(f"Recovered object_points shape: {object_points.shape}")
    print(f"Output format: {output_format}")
    print(f"Reconstruct endpoints: {reconstruct_endpoints}")
    print(f"Unflip z: {unflip_z}")
    if output_format == "dict":
        print(f"Dict key for object points: {dict_key}")
        if controller_points is not None:
            print(f"Included controller points under key: {controller_key}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert NPZ back to PKL using the inverse assumptions of final_pkl_to_npz.py."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input .npz file")
    parser.add_argument("--output", default=None, help="Path to output .pkl file")
    parser.add_argument(
        "--position-key",
        default="position",
        help="NPZ key containing aligned object points [T, N, 3]",
    )
    parser.add_argument(
        "--velocity-key",
        default="velocity",
        help="NPZ key containing aligned velocity [T, N, 3]",
    )
    parser.add_argument(
        "--no-unflip-z",
        action="store_true",
        help="Do not invert z back (by default z is unflipped).",
    )
    parser.add_argument(
        "--output-format",
        choices=["dict", "array"],
        default="dict",
        help="PKL payload type: dict for final.pkl-style, array for inference.pkl-style.",
    )
    parser.add_argument(
        "--dict-key",
        default="object_points",
        help="Object point key when --output-format dict.",
    )
    parser.add_argument(
        "--reconstruct-endpoints",
        choices=["none", "prepend", "append", "both"],
        default="none",
        help=(
            "Optionally reconstruct missing boundary frames from velocity. "
            "Forward conversion drops boundaries, so reconstruction is approximate for prepend."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step used when reconstructing endpoints from velocity.",
    )
    parser.add_argument(
        "--controller-data",
        default=None,
        help="Optional .pkl/.kpl file to copy controller points from.",
    )
    parser.add_argument(
        "--controller-key",
        default="controller_points",
        help="Controller key for output dict and lookup in --controller-data.",
    )
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help=("Optional max frame count to truncate both object and controller points to. "
              "Useful if controller data is much longer than object data and you want to avoid repeating.")
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)

    if args.output is None:
        root, _ = os.path.splitext(input_path)
        output_path = root + ".pkl"
    else:
        output_path = args.output

    controller_points = None
    if args.controller_data is not None:
        if not os.path.exists(args.controller_data):
            print(f"Controller data file does not exist: {args.controller_data}")
            sys.exit(1)
        data = load_kpl_or_pkl(args.controller_data)
        controller_points = _extract_controller_points(data, args.controller_key)

    convert(
        input_path=input_path,
        output_path=output_path,
        position_key=args.position_key,
        velocity_key=args.velocity_key,
        unflip_z=not args.no_unflip_z,
        output_format=args.output_format,
        dict_key=args.dict_key,
        reconstruct_endpoints=args.reconstruct_endpoints,
        dt=args.dt,
        controller_points=controller_points,
        controller_key=args.controller_key,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()