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
