"""Save a simple static 3D plot of PhysTwin controller points."""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqtt.utils.controller_collider import save_controller_point_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to final_data.pkl")
    parser.add_argument("output_path", help="Output PNG path")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--dense", action="store_true", help="Plot dense controller points")
    parser.add_argument("--hollow", action="store_true", help="Plot the hollow dense shell")
    parser.add_argument("--voxel-size", type=float, default=0.003)
    args = parser.parse_args()

    with open(args.data_path, "rb") as handle:
        data = pickle.load(handle)
    key = "controller_points_dense" if args.dense else "controller_points"
    if key not in data:
        raise KeyError(f"Dataset does not contain {key}")
    save_controller_point_plot(
        data[key],
        args.output_path,
        frame_index=args.frame,
        hollow=args.hollow,
        voxel_size=args.voxel_size,
    )
    print(f"Saved {key} plot to {args.output_path}")


if __name__ == "__main__":
    main()
