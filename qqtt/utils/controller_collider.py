"""Utilities for validating dense controller point clouds as colliders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


def count_visible_controller_frames(
    visibilities: np.ndarray,
    min_fraction: float,
) -> tuple[np.ndarray, int]:
    """Select controller points observed in at least ``min_fraction`` of frames.

    Returns ``(keep, min_visible)``: a boolean mask over points, and the frame
    count the threshold resolves to.

    A majority-visible point has every occluded gap bracketed by real
    measurements, so the fill can interpolate across it. A point seen only
    once is mostly CoTracker extrapolation and can reach through the object.
    """
    visibilities = np.asarray(visibilities).astype(bool)
    if visibilities.ndim != 2:
        raise ValueError(
            f"visibilities must be (frames, points), got {visibilities.shape}"
        )
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")

    num_frames = visibilities.shape[0]
    min_visible = int(np.ceil(min_fraction * num_frames))
    keep = np.count_nonzero(visibilities, axis=0) >= min_visible
    return keep, min_visible


def controller_depth_consistency(
    points: np.ndarray,
    masks: list,
    intrinsics: np.ndarray,
    w2cs: np.ndarray,
    depth_provider: Callable[[int, int], np.ndarray],
    depth_tolerance: float = 0.004,
    min_support_fraction: float = 0.7,
    max_conflict_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep tracks supported by measured claw depth without free-space conflicts.

    A camera supports a point when its projection hits the controller mask and
    measured depth agrees within ``depth_tolerance``. A closer measured surface
    is treated as occlusion; it is not evidence against the point. A point is
    contradicted when it lies in front of measured depth, or agrees with depth
    outside the controller mask. Tracks are retained if enough frames have
    direct support and conflicts occur in no more than the configured fraction.

    ``depth_provider(frame, camera)`` must return that camera's depth image in
    metres. Returns the per-track keep mask and support/conflict frame rates.
    """
    points = np.asarray(points, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    w2cs = np.asarray(w2cs, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points must be (frames, points, 3), got {points.shape}")
    if not 0.0 <= min_support_fraction <= 1.0:
        raise ValueError("min_support_fraction must be in [0, 1]")
    if not 0.0 <= max_conflict_fraction <= 1.0:
        raise ValueError("max_conflict_fraction must be in [0, 1]")
    if depth_tolerance < 0.0:
        raise ValueError("depth_tolerance must be non-negative")

    num_frames, num_points = points.shape[:2]
    if len(masks) != num_frames:
        raise ValueError(f"expected {num_frames} mask frames, got {len(masks)}")
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must be (cameras, 3, 3), got {intrinsics.shape}")
    if w2cs.shape != (len(intrinsics), 4, 4):
        raise ValueError(f"w2cs must be ({len(intrinsics)}, 4, 4), got {w2cs.shape}")

    supported = np.zeros((num_frames, num_points), dtype=bool)
    contradicted = np.zeros_like(supported)
    homogeneous = np.ones((num_points, 1), dtype=np.float64)
    finite = np.isfinite(points).all(axis=2)

    for frame in range(num_frames):
        for camera, camera_masks in sorted(masks[frame].items()):
            controller_mask = np.asarray(camera_masks["controller"], dtype=bool)
            height, width = controller_mask.shape
            camera_points = (w2cs[camera] @ np.c_[points[frame], homogeneous].T).T[:, :3]
            in_front = finite[frame] & (camera_points[:, 2] > 1e-6)
            if not in_front.any():
                continue

            projected = (intrinsics[camera] @ camera_points[in_front].T).T
            uv = projected[:, :2] / projected[:, 2:3]
            on_screen = (
                np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            )
            point_indices = np.flatnonzero(in_front)[on_screen]
            if not point_indices.size:
                continue

            pixels = np.floor(uv[on_screen]).astype(np.int64)
            measured_depth = np.asarray(depth_provider(frame, camera))
            if measured_depth.shape != controller_mask.shape:
                raise ValueError(
                    f"depth shape {measured_depth.shape} does not match mask "
                    f"shape {controller_mask.shape} at frame={frame}, camera={camera}"
                )
            observed_z = measured_depth[pixels[:, 1], pixels[:, 0]]
            has_depth = np.isfinite(observed_z) & (observed_z > 0)
            if not has_depth.any():
                continue

            point_indices = point_indices[has_depth]
            pixels = pixels[has_depth]
            observed_z = observed_z[has_depth]
            candidate_z = camera_points[point_indices, 2]
            in_controller = controller_mask[pixels[:, 1], pixels[:, 0]]
            residual = candidate_z - observed_z

            supported[frame, point_indices] |= (
                in_controller & (np.abs(residual) <= depth_tolerance)
            )
            contradicted[frame, point_indices] |= (
                (residual < -depth_tolerance)
                | ((np.abs(residual) <= depth_tolerance) & ~in_controller)
            )

    support_fraction = supported.mean(axis=0)
    conflict_fraction = contradicted.mean(axis=0)
    keep = (support_fraction >= min_support_fraction) & (
        conflict_fraction <= max_conflict_fraction
    )
    return keep, support_fraction, conflict_fraction


def fill_occluded_controller_points(
    points: np.ndarray,
    visibilities: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Rebuild controller positions on frames where a point was not observed.

    ``filter_track`` allocates ``track_points`` as zeros and writes only the
    frames CoTracker reported visible, so an occluded frame holds a zero, and a
    zero in the collider sits at the world origin. The gripper is rigid and
    moves smoothly (it rotates ~1.5 deg across a whole lift), so those frames
    are recoverable from the observations either side of the gap:

    * both sides observed -- linear interpolation between them,
    * only one side observed -- hold that measurement. This clamps to a real
      observation and never extrapolates through the occluder, which is what
      would let the collider reach past the object and grip through it.

    Which points are worth keeping is decided by
    ``count_visible_controller_frames``.

    Returns ``(filled, never_visible)``. Frames are interpolated or clamped;
    a point with no observation in any frame is left as NaN so the caller can
    drop it, and counted in ``never_visible``.
    """
    points = np.asarray(points, dtype=np.float64)
    visibilities = np.asarray(visibilities).astype(bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points must be (frames, points, 3), got {points.shape}")
    num_frames, num_points = visibilities.shape
    if points.shape[:2] != (num_frames, num_points):
        raise ValueError(
            f"points {points.shape[:2]} do not match visibilities "
            f"{(num_frames, num_points)}"
        )

    frames = np.arange(num_frames)
    # Nearest observed frame at or before / at or after each (frame, point).
    previous = np.maximum.accumulate(
        np.where(visibilities, frames[:, None], -1), axis=0
    )
    following = np.minimum.accumulate(
        np.where(visibilities, frames[:, None], num_frames)[::-1], axis=0
    )[::-1]

    out = points.copy()
    never = (previous < 0) & (following >= num_frames)
    out[never] = np.nan

    trailing = (previous >= 0) & (following >= num_frames) & ~visibilities
    if trailing.any():
        out[trailing] = points[previous[trailing], np.nonzero(trailing)[1]]

    leading = (previous < 0) & (following < num_frames) & ~visibilities
    if leading.any():
        out[leading] = points[following[leading], np.nonzero(leading)[1]]

    between = (previous >= 0) & (following < num_frames) & ~visibilities
    if between.any():
        fi, pi = np.nonzero(between)
        lo, hi = previous[between], following[between]
        span = np.maximum((hi - lo).astype(np.float64), 1.0)
        weight = (fi - lo) / span
        start = points[lo, pi]
        out[between] = start + (points[hi, pi] - start) * weight[:, None]

    return out, int(never.sum())


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
