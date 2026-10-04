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


def points_inside_any_mask(
    points: np.ndarray,
    masks: list,
    intrinsics: np.ndarray,
    w2cs: np.ndarray,
    mask_key: str = "controller",
) -> np.ndarray:
    """Mark (frame, point) pairs whose projection falls inside any camera mask.

    Relaxing the visibility threshold admits tracks whose occluded frames are
    reconstructed rather than observed, and some of those land in empty space
    between the gripper fingers -- where segmentation says nothing is present.
    A point outside every camera's mask on a frame is therefore treated as
    drifting, not as a genuine reconstruction.

    Returns a boolean array shaped like ``points`` minus the coordinate axis.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points must be (frames, points, 3), got {points.shape}")
    if not masks:
        return np.ones(points.shape[:2], dtype=bool)

    num_frames, num_points = points.shape[:2]
    inside = np.zeros((num_frames, num_points), dtype=bool)
    ones = np.ones((num_points, 1))
    finite = np.isfinite(points).all(axis=2)

    for frame in range(num_frames):
        # masks[frame] maps camera index -> {"object": ..., "controller": ...}
        for cam, cam_masks in sorted(masks[frame].items()):
            mask = np.asarray(cam_masks[mask_key])
            height, width = mask.shape
            cam_pts = w2cs[cam] @ np.concatenate(
                [points[frame], ones], axis=1
            ).T
            cam_pts = cam_pts.T[:, :3]
            valid = finite[frame] & (cam_pts[:, 2] > 1e-6)
            if not valid.any():
                continue
            uv = (intrinsics[cam] @ cam_pts[valid].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            on_screen = (
                (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            )
            indices = np.nonzero(valid)[0][on_screen]
            if indices.size == 0:
                continue
            pixels = np.floor(uv[on_screen]).astype(np.int64)
            hit = mask[pixels[:, 1], pixels[:, 0]]
            inside[frame, indices[hit]] = True

    return inside


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
