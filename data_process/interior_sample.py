"""Populate interior object points directly from masks and depth — no mesh.

The shape prior's only volumetric consumer is interior-point sampling in
``data_process_sample.py``, which previously required a watertight mesh so
``trimesh.sample.volume_mesh`` could ray-test containment. This module skips
the mesh entirely. It carves a voxel occupancy grid with the rule

    occupied(v) = inside the object mask in every camera
                  AND not in front of the observed surface in any camera,

erodes it one voxel from the free-space boundary so every surviving voxel is
strictly interior, and writes one jittered point per kept voxel (subsampled
to ``--max_points``). Everything below the support surface (``z > --floor_z``,
which is solid ground in this dataset's convention) is treated as free space
before carving, so no point ever ends up inside the table:

    {output_dir}/interior_points.npy
    {output_dir}/interior_points.ply   (for inspection)
    {output_dir}/interior_points.mp4   (turntable: carved hull + interior points)
    {output_dir}/interior_stats.json   (numbers, see summarize())

Key quality numbers reported by ``summarize()``:
- ``interior_fill_ratio``: interior volume / occupied volume. Low values mean
  the occupancy is a shell or wispy, i.e. the object is thin at this resolution.
- ``blobs`` / ``largest_blob_fraction``: number of separate interior regions and
  how much of the interior the biggest one holds (1.0 = single solid blob).
- ``mean_observed_distance``: how far the interior sits from the observed
  surface. Large values mean the interior is mostly *guessed* volume in regions
  no camera saw.
- ``observed_without_interior``: fraction of observed surface points with no
  interior point within ``--support_radius``. These are thin parts (fingers,
  rope, cloth edges) that cannot hold interior samples.
- ``points_below_floor``: must always be 0; a non-zero value means points are
  being generated underneath the support surface.

Thin geometry (cloth, rope) has no interior at voxel resolution and yields
zero points, which downstream treats as an empty interior set.
"""

import json
import os
import pickle
import sys
from argparse import ArgumentParser

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

try:
    # interior_sample.py is launched directly by process_data.py, so
    # data_process is the script directory rather than an importable package.
    from controller_labels import is_controller_label, parse_controller_names
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_process.controller_labels import (
        is_controller_label,
        parse_controller_names,
    )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to {base_path}/{case_name}/shape.",
    )
    parser.add_argument("--controller_names", type=str, default=None)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.004 / 3,
        help="Voxel edge length in metres (default: 1.33 mm).",
    )
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=12_000_000,
        help="Safety cap on the occupancy grid size; voxel_size is grown to fit.",
    )
    parser.add_argument("--bbox_pad", type=float, default=0.02)
    parser.add_argument(
        "--outlier_voxel",
        type=float,
        default=None,
        help="Coarse voxel size for dropping sparse observed-point outliers; "
        "defaults to max(4 * --voxel_size, 0.016).",
    )
    parser.add_argument(
        "--mask_erode",
        type=int,
        default=3,
        help="Pixels to erode each object mask before carving.",
    )
    parser.add_argument(
        "--depth_margin",
        type=float,
        default=0.004,
        help="Voxels closer than this to the observed surface are not carved.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=None,
        help="Divide raw depth by this to reach metres. Integer depth defaults "
        "to 1000 (millimetres); float depth is assumed to be metres.",
    )
    parser.add_argument("--depth_max", type=float, default=2.0)
    parser.add_argument(
        "--floor_z",
        type=float,
        default=0.0,
        help="World z of the support surface. Nothing is carved below it, "
        "i.e. z > floor_z is solid ground (the convention of this dataset, "
        "where the floor is z=0 and z increases downwards).",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=10_000,
        help="Maximum number of interior points to write (random subsample).",
    )
    parser.add_argument(
        "--support_radius",
        type=float,
        default=0.01,
        help="Radius in metres used to report how much observed surface is "
        "backed by interior samples (diagnostics only).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip the interior_points.mp4 turntable diagnostic.",
    )
    return parser.parse_args()


def object_index(case_dir, cam, controller_names):
    info_path = f"{case_dir}/mask/mask_info_{cam}.json"
    with open(info_path, "r") as handle:
        data = json.load(handle)
    obj_idx = None
    for key, value in data.items():
        if not is_controller_label(value, controller_names):
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    if obj_idx is None:
        raise ValueError(f"No object detection found in {info_path}.")
    return obj_idx


def load_depth(path, depth_scale):
    raw = np.load(path)
    if np.issubdtype(raw.dtype, np.integer):
        scale = 1000.0 if depth_scale is None else depth_scale
        return raw.astype(np.float32) / scale
    return raw.astype(np.float32)


def backproject_masked(depth_m, mask, intrinsic, c2w, depth_max):
    ys, xs = np.nonzero(mask)
    z = depth_m[ys, xs]
    valid = (z > 0) & (z < depth_max)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    cam = np.stack(
        [
            (xs - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (ys - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ],
        axis=1,
    ).astype(np.float64)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def largest_cluster(points, coarse):
    """Drop sparse silhouette/depth outliers by keeping the largest
    6-connected component of a coarse voxelization."""
    if points.size == 0:
        return points
    low = points.min(axis=0)
    dims = np.floor((points.max(axis=0) - low) / coarse).astype(int) + 1
    indices = np.floor((points - low) / coarse).astype(int)
    grid = np.zeros(dims, dtype=bool)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(grid, structure=structure)
    if count <= 1:
        return points
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, count + 1))
    best = int(np.argmax(sizes)) + 1
    keep = labels[indices[:, 0], indices[:, 1], indices[:, 2]] == best
    return points[keep]


def carve_occupancy(dims, grid_origin, voxel, masks, depths, intrinsics, w2cs,
                    depth_margin, floor_z):
    nx, ny, nz = dims
    inside_all = np.ones((nx, ny, nz), dtype=bool)
    free_any = np.zeros((nx, ny, nz), dtype=bool)

    # Bound peak memory by carving one slab of x-rows at a time.
    chunk = max(1, int(1_000_000 // max(1, ny * nz)))
    for i0 in range(0, nx, chunk):
        i1 = min(nx, i0 + chunk)
        x = grid_origin[0] + (np.arange(i0, i1) + 0.5) * voxel
        y = grid_origin[1] + (np.arange(ny) + 0.5) * voxel
        z = grid_origin[2] + (np.arange(nz) + 0.5) * voxel
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        inside = np.ones(centers.shape[0], dtype=bool)
        free = np.zeros(centers.shape[0], dtype=bool)
        for cam in range(len(w2cs)):
            w2c = w2cs[cam]
            cam_pts = centers @ w2c[:3, :3].T + w2c[:3, 3]
            depth = cam_pts[:, 2]
            front = depth > 1e-6
            safe_depth = np.where(front, depth, 1.0)
            u = (intrinsics[cam][0, 0] * cam_pts[:, 0] / safe_depth
                 + intrinsics[cam][0, 2])
            v = (intrinsics[cam][1, 1] * cam_pts[:, 1] / safe_depth
                 + intrinsics[cam][1, 2])
            ui = np.floor(u).astype(np.int64)
            vi = np.floor(v).astype(np.int64)
            h, w = masks[cam].shape
            inb = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

            cam_inside = np.zeros(centers.shape[0], dtype=bool)
            cam_inside[inb] = masks[cam][vi[inb], ui[inb]]
            inside &= cam_inside

            observed = np.zeros(centers.shape[0], dtype=np.float32)
            observed[inb] = depths[cam][vi[inb], ui[inb]]
            free |= inb & (observed > 0) & (depth < observed - depth_margin)

        inside_all[i0:i1] = inside.reshape(i1 - i0, ny, nz) & (z <= floor_z)
        free_any[i0:i1] = free.reshape(i1 - i0, ny, nz)

    return inside_all & ~free_any


def erode_to_interior(occupied):
    """Remove voxels within one 26-neighbourhood step of free space."""
    free = ~occupied
    near_free = ndimage.binary_dilation(
        free, structure=ndimage.generate_binary_structure(3, 3)
    )
    return occupied & ~near_free


def sample_interior_points(interior, grid_origin, voxel, max_points, rng):
    """One uniformly jittered point per kept voxel, subsampled to max_points."""
    indices = np.argwhere(interior)
    if indices.shape[0] > max_points:
        chosen = rng.choice(indices.shape[0], size=max_points, replace=False)
        indices = indices[chosen]
    jitter = rng.random(indices.shape)
    return grid_origin[None, :] + (indices + jitter) * voxel


def render_diagnostic(output_dir, occupied, grid_origin, voxel, points):
    """Turntable video of the carved hull with the interior points inside it."""
    verts, faces, _, _ = measure.marching_cubes(
        occupied.astype(np.float32), level=0.5
    )
    hull = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts * voxel + grid_origin),
        o3d.utility.Vector3iVector(faces),
    )
    # Cut the lower half of the hull away so the interior points stay visible.
    z_cut = float(np.median(points[:, 2])) if points.size else 0.0
    hull.remove_triangles_by_mask(
        np.asarray(hull.vertices)[np.asarray(hull.triangles)][:, :, 2].mean(axis=1) <= z_cut
    )
    hull.paint_uniform_color([0.75, 0.75, 0.75])
    interior_pcd = o3d.geometry.PointCloud()
    interior_pcd.points = o3d.utility.Vector3dVector(points)
    interior_pcd.paint_uniform_color([0.85, 0.1, 0.1])

    # Render a normalised copy so the framing does not depend on the object
    # size: Open3D's default camera has no auto-fit.
    box = hull.get_axis_aligned_bounding_box()
    for geometry in (hull, interior_pcd):
        geometry.scale(1.0 / max(box.get_extent().max(), 1e-6), box.get_center())
        geometry.translate(-box.get_center())

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    video_writer = cv2.VideoWriter(
        f"{output_dir}/interior_points.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    vis.add_geometry(hull)
    vis.add_geometry(interior_pcd)
    view_control = vis.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    for _ in range(36):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        video_writer.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vis.destroy_window()


def summarize(interior, points, occupied, dims, voxel, observed, support_radius,
              floor_z):
    """Diagnostics for judging interior-fill quality (see module docstring)."""
    obs_low, obs_high = observed.min(axis=0), observed.max(axis=0)
    stats = {
        "voxel_size": float(voxel),
        "grid_dims": [int(d) for d in dims],
        "occupied_voxels": int(occupied.sum()),
        "interior_voxels": int(interior.sum()),
        "occupied_volume_m3": float(occupied.sum()) * voxel ** 3,
        "interior_volume_m3": float(interior.sum()) * voxel ** 3,
        "interior_fill_ratio": (
            float(interior.sum() / occupied.sum()) if occupied.any() else 0.0
        ),
        "points": int(points.shape[0]),
        "points_below_floor": int((points[:, 2] > floor_z).sum()),
        "observed_bbox_extent_m": (obs_high - obs_low).round(4).tolist(),
    }
    if not interior.any():
        stats.update(
            {
                "interior_bbox_extent_m": [0.0, 0.0, 0.0],
                "blobs": 0,
                "largest_blob_fraction": 0.0,
                "mean_observed_distance": 0.0,
                "observed_without_interior": 1.0,
            }
        )
        return stats

    labels, count = ndimage.label(interior, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())[1:]
    stats["interior_bbox_extent_m"] = (
        points.max(axis=0) - points.min(axis=0)
    ).round(4).tolist()
    stats["blobs"] = int(count)
    stats["largest_blob_fraction"] = float(sizes.max() / sizes.sum())

    tree = cKDTree(observed)
    # Distance from each interior point to the nearest observed surface point:
    # large values are volume no camera actually saw.
    stats["mean_observed_distance"] = float(
        tree.query(points)[0].mean().round(4)
    )
    covered = tree.query_ball_point(
        observed, support_radius, return_length=True, workers=-1
    ) > 0
    stats["observed_without_interior"] = float(1.0 - covered.mean())
    return stats


def print_summary(output_dir, stats):
    print(f"[interior_sample] {output_dir}/interior_points.npy")
    print(
        f"  voxel={stats['voxel_size'] * 1000:.2f}mm "
        f"grid={stats['grid_dims']} "
        f"occupied={stats['occupied_voxels']} "
        f"interior={stats['interior_voxels']} "
        f"fill_ratio={stats['interior_fill_ratio']:.3f}"
    )
    print(
        f"  points={stats['points']} below_floor={stats['points_below_floor']} "
        f"blobs={stats['blobs']} "
        f"largest_blob={stats['largest_blob_fraction']:.3f} "
        f"interior_volume={stats['interior_volume_m3'] * 1e3:.2f}L"
    )
    print(
        f"  observed_bbox_extent={stats['observed_bbox_extent_m']} "
        f"interior_bbox_extent={stats['interior_bbox_extent_m']}"
    )
    print(
        f"  mean_dist_to_observed={stats['mean_observed_distance'] * 1000:.1f}mm "
        f"observed_without_interior={stats['observed_without_interior']:.3f}"
    )


def main():
    args = parse_args()
    case_dir = f"{args.base_path}/{args.case_name}"
    output_dir = args.output_dir or f"{case_dir}/shape"
    os.makedirs(output_dir, exist_ok=True)
    controller_names = parse_controller_names(args.controller_names)

    with open(f"{case_dir}/metadata.json", "r") as handle:
        metadata = json.load(handle)
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    num_cam = len(intrinsics)
    with open(f"{case_dir}/calibrate.pkl", "rb") as handle:
        c2ws = pickle.load(handle)

    masks, depths, w2cs, observed = [], [], [], []
    for cam in range(num_cam):
        obj_idx = object_index(case_dir, cam, controller_names)
        mask = imageio.imread(f"{case_dir}/mask/{cam}/{obj_idx}/{args.frame}.png")
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask > 127
        if args.mask_erode > 0:
            mask = ndimage.binary_erosion(mask, iterations=args.mask_erode)
        depth = load_depth(
            f"{case_dir}/depth/{cam}/{args.frame}.npy", args.depth_scale
        )
        masks.append(mask)
        depths.append(depth)
        w2cs.append(np.linalg.inv(c2ws[cam]))
        observed.append(
            backproject_masked(depth, mask, intrinsics[cam], c2ws[cam], args.depth_max)
        )

    observed = np.concatenate([points for points in observed if points.size])
    if observed.size == 0:
        raise RuntimeError("No observed object points; check masks and depth.")
    # Points below the support surface are depth noise inside the table.
    observed = observed[observed[:, 2] <= args.floor_z]

    coarse = args.outlier_voxel or max(4 * args.voxel_size, 0.016)
    observed = largest_cluster(observed, coarse)
    print(
        f"[interior_sample] observed object points after outlier filtering: "
        f"{observed.shape[0]}"
    )

    low = observed.min(axis=0) - args.bbox_pad
    high = observed.max(axis=0) + args.bbox_pad
    extent = high - low

    voxel = args.voxel_size
    dims = np.ceil(extent / voxel).astype(int) + 3
    while int(np.prod(dims)) > args.max_voxels:
        voxel *= 1.25
        dims = np.ceil(extent / voxel).astype(int) + 3
    if not np.isclose(voxel, args.voxel_size):
        print(
            f"[interior_sample] voxel_size grown to {voxel:.4f} m to fit "
            f"--max_voxels={args.max_voxels}"
        )
    dims = tuple(int(d) for d in dims)
    grid_origin = low - voxel
    print(f"[interior_sample] grid dims={dims} voxel={voxel:.4f} m")

    occupied = carve_occupancy(
        dims, grid_origin, voxel, masks, depths, intrinsics, w2cs, args.depth_margin,
        args.floor_z,
    )

    # Force-occupy every voxel containing an observed point so no observed
    # surface voxel is treated as free space.
    indices = np.floor((observed - grid_origin) / voxel).astype(int)
    valid = np.all(indices >= 0, axis=1) & np.all(
        indices < np.asarray(dims), axis=1
    )
    indices = indices[valid]
    occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    interior = erode_to_interior(occupied)
    rng = np.random.default_rng(args.seed)
    points = sample_interior_points(interior, grid_origin, voxel, args.max_points, rng)

    np.save(f"{output_dir}/interior_points.npy", points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(f"{output_dir}/interior_points.ply", pcd)

    stats = summarize(
        interior, points, occupied, dims, voxel, observed, args.support_radius,
        args.floor_z,
    )
    with open(f"{output_dir}/interior_stats.json", "w") as handle:
        json.dump(stats, handle, indent=2)
    print_summary(output_dir, stats)

    if not args.no_visualize:
        render_diagnostic(output_dir, occupied, grid_origin, voxel, points)
        print(f"[interior_sample] {output_dir}/interior_points.mp4")


if __name__ == "__main__":
    main()
