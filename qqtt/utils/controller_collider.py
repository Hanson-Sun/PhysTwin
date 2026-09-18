"""Utilities for validating dense controller point clouds as colliders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ControllerColliderCoverage:
    """Summary of point spacing and object proximity over a trajectory."""

    frame_count: int
    point_count: int
    max_nearest_neighbor_spacing: float
    p95_nearest_neighbor_spacing: float
    median_nearest_neighbor_spacing: float
    recommended_radius_for_1_5x_overlap: float
    min_object_distance: float | None
    first_object_distance_frame: int | None


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, points, 3), got {points.shape}")
    if points.shape[0] == 0 or points.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two points in one frame")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return points


def hollow_controller_points(
    controller_points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Remove duplicate and fully enclosed controller voxels.

    Selection is based on frame zero and applied to every frame so the output
    remains a consistent tracked trajectory. A voxel is retained when it lies
    on the six-connected boundary of the occupied voxel set.
    """
    points = _as_points(controller_points, "controller_points")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    voxel_coords = np.floor(points[0] / voxel_size).astype(np.int64)
    unique_coords, unique_indices = np.unique(
        voxel_coords, axis=0, return_index=True
    )
    occupied = {tuple(coord) for coord in unique_coords}
    shell_mask = np.zeros(len(unique_coords), dtype=bool)
    neighbor_offsets = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    for index, coord in enumerate(unique_coords):
        coord_tuple = tuple(coord)
        shell_mask[index] = any(
            tuple(coord + offset) not in occupied for offset in neighbor_offsets
        )

    selected_indices = unique_indices[shell_mask]
    if selected_indices.size < 2:
        raise ValueError(
            "Hollow controller filtering removed too many points; "
            f"input={points.shape[1]}, output={selected_indices.size}, "
            f"voxel_size={voxel_size}"
        )
    return points[:, selected_indices, :].copy()


def save_controller_point_plot(
    controller_points: np.ndarray,
    output_path: str,
    frame_index: int = 0,
    hollow: bool = False,
    voxel_size: float = 0.003,
) -> None:
    """Save a simple 3D scatter plot of one controller trajectory frame."""
    import matplotlib.pyplot as plt

    points = _as_points(controller_points, "controller_points")
    if frame_index < 0 or frame_index >= points.shape[0]:
        raise IndexError(f"frame_index must be in [0, {points.shape[0]})")
    frame_points = points
    if hollow:
        frame_points = hollow_controller_points(points, voxel_size)
    frame_points = frame_points[frame_index]

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(frame_points[:, 0], frame_points[:, 1], frame_points[:, 2], s=2)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(f"Controller points, frame {frame_index}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def measure_controller_collider_coverage(
    controller_points: np.ndarray,
    object_points: np.ndarray | None = None,
    contact_distance: float | None = None,
) -> ControllerColliderCoverage:
    """Measure dense-point spacing and optional proximity to object points.

    The recommended sphere radius uses a conservative 1.5x overlap rule:
    nearest-neighbor spacing must be no greater than 1.5 times the radius.
    """
    controller_points = _as_points(controller_points, "controller_points")
    if object_points is not None:
        object_points = _as_points(object_points, "object_points")
        if object_points.shape[0] != controller_points.shape[0]:
            raise ValueError("controller_points and object_points must have the same frame count")

    nearest_neighbor_distances = []
    object_distances = []
    for frame_idx, frame_points in enumerate(controller_points):
        tree = cKDTree(frame_points)
        nearest_neighbor_distances.append(tree.query(frame_points, k=2)[0][:, 1])
        if object_points is not None:
            object_tree = cKDTree(object_points[frame_idx])
            object_distances.append(object_tree.query(frame_points, k=1)[0].min())

    spacing = np.concatenate(nearest_neighbor_distances)
    min_object_distance = None
    first_object_distance_frame = None
    if object_distances:
        object_distances = np.asarray(object_distances)
        min_object_distance = float(object_distances.min())
        if contact_distance is not None:
            contact_frames = np.flatnonzero(object_distances <= contact_distance)
            if contact_frames.size:
                first_object_distance_frame = int(contact_frames[0])

    max_spacing = float(spacing.max())
    return ControllerColliderCoverage(
        frame_count=int(controller_points.shape[0]),
        point_count=int(controller_points.shape[1]),
        max_nearest_neighbor_spacing=max_spacing,
        p95_nearest_neighbor_spacing=float(np.percentile(spacing, 95)),
        median_nearest_neighbor_spacing=float(np.median(spacing)),
        recommended_radius_for_1_5x_overlap=max_spacing / 1.5,
        min_object_distance=min_object_distance,
        first_object_distance_frame=first_object_distance_frame,
    )
