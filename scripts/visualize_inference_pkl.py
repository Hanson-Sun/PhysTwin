#!/usr/bin/env python3
"""
Simple interactive viewer for inference.pkl trajectories.

Controls:
- Right / n: next frame
- Left / p: previous frame
- Home: first frame
- End: last frame
- q or Esc: quit
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(path, key):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if key in data:
            traj = np.asarray(data[key], dtype=np.float32)
        elif "object_points" in data:
            traj = np.asarray(data["object_points"], dtype=np.float32)
        elif "controller_points" in data:
            traj = np.asarray(data["controller_points"], dtype=np.float32)
        else:
            keys = ", ".join(sorted(str(k) for k in data.keys()))
            raise KeyError(f"No usable trajectory key found. Available keys: {keys}")
    else:
        traj = np.asarray(data, dtype=np.float32)

    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(f"Expected [T, N, 3] trajectory, got {traj.shape}")

    return traj


def set_equal_3d_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max()) * 0.55
    if radius <= 1e-8:
        radius = 0.05

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main():
    parser = argparse.ArgumentParser(description="Visualize points from inference.pkl")
    parser.add_argument("--input", required=True, help="Path to inference.pkl")
    parser.add_argument(
        "--key",
        default="object_points",
        help="Dict key to use when PKL payload is a dict",
    )
    parser.add_argument("--frame", type=int, default=0, help="Initial frame index")
    parser.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="Max points to draw per frame (<=0 means all points)",
    )
    args = parser.parse_args()

    traj = load_trajectory(args.input, args.key)
    num_frames, num_points, _ = traj.shape

    sample_idx = None
    if args.max_points > 0 and num_points > args.max_points:
        sample_idx = np.linspace(0, num_points - 1, args.max_points, dtype=int)

    state = {"frame": int(np.clip(args.frame, 0, num_frames - 1))}

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def draw():
        ax.clear()
        pts = traj[state["frame"]]
        if sample_idx is not None:
            pts = pts[sample_idx]

        valid = ~np.isnan(pts).any(axis=1)
        pts = pts[valid]
        if len(pts) == 0:
            pts = np.zeros((1, 3), dtype=np.float32)

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="tab:blue", alpha=0.85)
        set_equal_3d_axes(ax, pts)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Frame {state['frame'] + 1}/{num_frames} | points: {len(pts)}\n"
            "Left/Right or p/n to navigate, q to quit"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "n"):
            state["frame"] = (state["frame"] + 1) % num_frames
            draw()
        elif event.key in ("left", "p"):
            state["frame"] = (state["frame"] - 1) % num_frames
            draw()
        elif event.key == "home":
            state["frame"] = 0
            draw()
        elif event.key == "end":
            state["frame"] = num_frames - 1
            draw()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
