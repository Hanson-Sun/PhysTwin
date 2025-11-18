"""
depth_scale_align.py
====================

Tools to help align and scale depth maps produced independently per camera (e.g. from Video-Depth-Anything).

Problem: per-view depth estimators often output depths with unknown scale and bias per camera
and do not enforce multi-view consistency. This script provides utilities to:
- project per-frame depth maps into world coordinates using intrinsics + extrinsics
- optionally apply per-camera corrections (uniform scale or affine scale+bias)
- optimize corrections using multi-frame point clouds and object masks
- perform ICP-based alignment and refinement
- visualize point clouds and camera poses

Usage patterns:
- Provide `--case_dir <case>` to auto-load `depth/0,1,2`, `calibrate.pkl`, `metadata.json`, and `mask/`.
- Use `--optimize_scales` with `--correction scale|affine` to fit per-camera corrections.
- Use `--method icp_refine|robust|scale` to pick alignment strategy.

Notes and assumptions:
- Depth bias `b` is in meters (the script normalizes mm->m when large values are detected).
- The affine correction model is z' = s * z + b applied in each camera's coordinate frame.
- The optimization minimizes symmetric nearest-neighbor distances between corrected per-camera
    point clouds; this requires sufficient overlap between cameras and benefits from masks.
"""

import os
import numpy as np
import open3d as o3d
import json
import pickle
import copy
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import cv2
import glob

def load_depth_and_project(depth_path, intrinsic, extrinsic, mask2d=None):
    """
    Load a depth map and project to 3D points in world coordinates.
    """
    depth = np.load(depth_path)
    # Convert depth from millimeters to meters if values are large
    if np.max(depth) > 100:  # Heuristic: if max > 10, likely in mm
        print("Converting depth from millimeters to meters.")
        depth = depth / 1000.0
    print(f"Loaded depth map from {depth_path}, shape: {depth.shape}, dtype: {depth.dtype}, min: {np.min(depth)}, max: {np.max(depth)}")
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be 2D, got shape {depth.shape}")
    h, w = depth.shape
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    print(f"Using intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"Extrinsic matrix (before inversion):\n{extrinsic}")
    # Create a grid of (u,v) pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # Project to camera coordinates
    z2d = depth
    valid2d = z2d > 0
    if mask2d is not None:
        try:
            mask2d = np.asarray(mask2d).astype(bool)
            if mask2d.shape == depth.shape:
                valid2d = np.logical_and(valid2d, mask2d)
            else:
                print(f"Warning: mask shape {mask2d.shape} does not match depth shape {depth.shape}; ignoring mask")
        except Exception as e:
            print(f"Warning: could not apply mask due to: {e}")

    z = z2d.flatten()
    u_f = u.flatten()
    v_f = v.flatten()
    # Project only valid pixels
    valid = valid2d.flatten()
    z = z[valid]
    u_f = u_f[valid]
    v_f = v_f[valid]
    x = (u_f - cx) * z / fx
    y = (v_f - cy) * z / fy
    points_cam = np.stack((x, y, z), axis=1)
    # Convert to homogeneous
    points_cam_h = np.concatenate([points_cam, np.ones((points_cam.shape[0],1))], axis=1)
    # Transform to world coordinates
    # NOTE: If extrinsic is world-to-camera, invert it to get camera-to-world
    if np.allclose(np.dot(extrinsic, np.linalg.inv(extrinsic)), np.eye(4), atol=1e-3):
        # Assume input is camera-to-world (default PhysTwin convention)
        extr = extrinsic
    else:
        # If not, invert (for safety, but this branch should rarely trigger)
        extr = np.linalg.inv(extrinsic)
        print(f"Extrinsic matrix was inverted. Now using:\n{extr}")
    print(f"Translation part of extrinsic: {extr[:3,3]}")
    points_world = (extr @ points_cam_h.T).T[:, :3]
    print(f"Projected {points_world.shape[0]} points to world coordinates. Example: {points_world[0] if points_world.shape[0] > 0 else 'None'}")
    return points_world

def np_to_o3d(points, color=None):
    points = np.asarray(points).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

def visualize_point_clouds(pcds, title, render_as_mesh=False):
    colors = [[1,0,0],[0,1,0],[0,0,1]]
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)

    for i, p in enumerate(pcds):
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(p.points))
        vis_pcd.paint_uniform_color(colors[i%3])
        if render_as_mesh:
            try:
                vis_pcd.estimate_normals()
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(vis_pcd, depth=8)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(colors[i%3])
                vis.add_geometry(mesh)
            except Exception:
                vis.add_geometry(vis_pcd)
        else:
            vis.add_geometry(vis_pcd)

    # Highlight all camera positions and directions
    try:
        import __main__
        if hasattr(__main__, 'c2ws'):
            cam_colors = [[1,1,0], [0,1,0], [0,0,1]]  # Yellow, Green, Blue
            for i, c2w in enumerate(__main__.c2ws):
                cam_extr = np.array(c2w)
                cam_pos = cam_extr[:3, 3]
                look_dir = cam_extr[:3, 2]
                cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                cam_sphere.translate(cam_pos)
                cam_sphere.paint_uniform_color(cam_colors[i])
                vis.add_geometry(cam_sphere)
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.08, cone_height=0.04)
                def rotation_matrix_from_vectors(vec1, vec2):
                    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
                    v = np.cross(a, b)
                    c = np.dot(a, b)
                    s = np.linalg.norm(v)
                    if s == 0:
                        return np.eye(3)
                    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
                rot = rotation_matrix_from_vectors(np.array([0, 0, 1]), look_dir)
                arrow.rotate(rot, center=np.array([0, 0, 0]))
                arrow.translate(cam_pos)
                arrow.paint_uniform_color([1, 0, 0])
                vis.add_geometry(arrow)
    except Exception as e:
        print(f"Could not visualize cameras: {e}")

    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    render_option.light_on = True

    # Set view to camera 0 pose if available
    try:
        import __main__
        if hasattr(__main__, 'c2ws'):
            cam0_extr = np.array(__main__.c2ws[0])
            ctr = vis.get_view_control()
            cam_pos = cam0_extr[:3, 3]
            look_dir = cam0_extr[:3, 2]
            lookat = cam_pos + look_dir * 0.5
            up = -cam0_extr[:3, 1]
            ctr.set_lookat(lookat)
            ctr.set_up(up)
            ctr.set_front(look_dir)
            ctr.set_zoom(0.5)
    except Exception as e:
        print(f"Could not set camera view: {e}")

    vis.run()
    vis.destroy_window()

def align_point_clouds_with_scale_robust(pcds, c2ws):
    """
    More robust alignment using camera poses as initial guesses.
    """
    aligned_pcds = [pcds[0]]
    transforms = [np.eye(4)]
    for i in range(1, len(pcds)):
        # Compute initial transformation from camera poses
        # c2ws[i] is camera-to-world for camera i
        # To align pcds[i] to pcds[0], we need the relative transform
        rel_transform = np.linalg.inv(c2ws[i]) @ c2ws[0]  # world-to-cam_i to cam_0-to-world
        # Actually, since points are already in world coords, but to align pcds[i] to pcds[0]
        # The initial guess is the inverse of the relative pose
        initial_guess = np.linalg.inv(rel_transform)
        
        # Downsample for robustness
        pcd_source = pcds[i].voxel_down_sample(voxel_size=0.01)
        pcd_target = aligned_pcds[0].voxel_down_sample(voxel_size=0.01)
        
        # ICP with scaling, using initial guess
        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target, 0.05, initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        )
        pcd_aligned = o3d.geometry.PointCloud(pcds[i])
        pcd_aligned.transform(reg.transformation)
        aligned_pcds.append(pcd_aligned)
        transforms.append(reg.transformation)
    return aligned_pcds, transforms

def align_point_clouds_with_scale(pcds):
    aligned_pcds = [pcds[0]]
    transforms = [np.eye(4)]
    for i in range(1, len(pcds)):
        reg = o3d.pipelines.registration.registration_icp(
            pcds[i], aligned_pcds[0], 0.05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        )
        pcd_aligned = o3d.geometry.PointCloud(pcds[i])
        pcd_aligned.transform(reg.transformation)
        aligned_pcds.append(pcd_aligned)
        transforms.append(reg.transformation)
    return aligned_pcds, transforms

def align_point_clouds_icp_refine(pcds, c2ws=None, voxel_size=0.01, max_corr_dist=0.05):
    """
    Refine alignment using point-to-plane ICP (if normals available).
    Uses camera poses (if provided) to form an initial guess per camera, otherwise uses identity.
    Returns aligned point clouds and the transforms applied to each.
    """
    aligned = [pcds[0]]
    transforms = [np.eye(4)]
    # Estimate normals for point-to-plane
    for p in pcds:
        try:
            p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        except Exception:
            pass

    for i in range(1, len(pcds)):
        source = pcds[i]
        target = aligned[0]

        # initial guess from camera poses if available
        init = np.eye(4)
        if c2ws is not None:
            try:
                rel_transform = np.linalg.inv(c2ws[i]) @ c2ws[0]
                init = np.linalg.inv(rel_transform)
            except Exception:
                init = np.eye(4)

        # Downsample for ICP
        src_down = source.voxel_down_sample(voxel_size=voxel_size)
        tgt_down = target.voxel_down_sample(voxel_size=voxel_size)
        try:
            src_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            tgt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        except Exception:
            pass

        # Try point-to-plane; fall back to point-to-point if fails
        try:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            reg = o3d.pipelines.registration.registration_icp(
                src_down, tgt_down, max_corr_dist, init, estimation)
        except Exception:
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            reg = o3d.pipelines.registration.registration_icp(
                src_down, tgt_down, max_corr_dist, init, estimation)

        T = reg.transformation
        p_aligned = o3d.geometry.PointCloud(pcds[i])
        p_aligned.transform(T)
        aligned.append(p_aligned)
        transforms.append(T)

    return aligned, transforms

def optimize_depth_scales(pcds, c2ws, sample_per_pcd=2000, bounds=(0.5, 3.0)):
    """
    Optimize per-camera uniform scale corrections.

    Method summary
    --------------
    - Each camera has a single scalar scale `s` applied about its camera center:
            p_scaled = cam_center + s * (p - cam_center)
        This is equivalent to scaling the depth values uniformly while keeping the camera center fixed.
    - The optimizer samples points from each camera's merged point cloud (to keep the problem small),
        applies the candidate scales to those samples, and measures the multi-view inconsistency
        by computing for each scaled point its nearest-neighbor distance to the union of the other
        scaled clouds. The objective is the mean nearest-neighbor distance (summed over cameras).
    - Optimization is performed using L-BFGS-B over the per-camera scale variables with box bounds.

    Assumptions and notes
    ---------------------
    - Camera-to-world poses (`c2ws`) must be correct; scales only change radial distance from each
        camera center and cannot correct pose errors.
    - Use object masks and multiple frames to reduce bias from large planar surfaces (tables, floors).
    - This method finds a single uniform scale per camera; it cannot correct additive depth bias.
    - The loss is non-convex; good initialization and sufficient overlap across cameras improves results.
    """
    n = len(pcds)
    cam_centers = [np.array(c2w)[:3, 3] for c2w in c2ws]

    # Sample points from each pcd to keep optimization fast
    samples = []
    for pcd in pcds:
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            samples.append(np.zeros((0,3)))
            continue
        if pts.shape[0] > sample_per_pcd:
            idx = np.random.choice(pts.shape[0], sample_per_pcd, replace=False)
            pts_s = pts[idx]
        else:
            pts_s = pts
        samples.append(pts_s)

    def apply_scales(scales):
        scaled = []
        for i in range(n):
            c = cam_centers[i]
            pts = samples[i]
            scaled_pts = c + scales[i] * (pts - c)
            scaled.append(scaled_pts)
        return scaled

    def loss_fn(x):
        scales = np.array(x)
        scaled = apply_scales(scales)
        # Build KD-trees for each set
        trees = [cKDTree(s) if s.shape[0] > 0 else None for s in scaled]
        total = 0.0
        # Symmetric NN: for each set, distance to union of others
        for i in range(n):
            src = scaled[i]
            if src.shape[0] == 0:
                continue
            # build union of others
            others = np.vstack([scaled[j] for j in range(n) if j != i and scaled[j].shape[0] > 0]) if any(j != i and scaled[j].shape[0] > 0 for j in range(n)) else np.empty((0,3))
            if others.shape[0] == 0:
                continue
            tree = cKDTree(others)
            dists, _ = tree.query(src, k=1)
            total += dists.mean()
        return total

    x0 = np.ones(n)
    bnds = [(bounds[0], bounds[1]) for _ in range(n)]
    res = minimize(loss_fn, x0, bounds=bnds, method='L-BFGS-B', options={'maxiter':200})
    scales_opt = res.x

    # Apply optimized scales to full point clouds
    scaled_pcds = []
    for i in range(n):
        pts_full = np.asarray(pcds[i].points)
        c = cam_centers[i]
        scaled_full = c + scales_opt[i] * (pts_full - c)
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(scaled_full.astype(np.float32))
        if pcds[i].has_colors():
            pcd_new.colors = pcds[i].colors
        scaled_pcds.append(pcd_new)

    return scales_opt, scaled_pcds


def optimize_depth_affine(pcds, c2ws, intrinsics, sample_per_pcd=2000, bounds_s=(0.5, 3.0), bounds_b=(-0.5, 0.5)):
    """
    Optimize per-camera affine depth corrections (scale + bias).

    Method summary
    --------------
    - Each camera has two parameters (s, b) defining an affine correction on depth values in that
      camera's coordinate frame: z' = s * z + b. The corrected depth z' is then used to scale the
      camera-space point: p_cam' = (x * (z'/z), y * (z'/z), z'). That ensures x,y are consistent
      with the corrected range.
    - The optimizer samples points from the merged per-camera point clouds, transforms them into
      the corresponding camera frame, applies the affine correction, and transforms them back to
      world coordinates. It then measures multi-view inconsistency by nearest-neighbor distances
      between the corrected clouds (symmetric NN loss) and minimizes the mean distance across cameras.
    - Optimization is done with L-BFGS-B over the 2*N parameters with box bounds for stability.

    Assumptions and notes
    ---------------------
    - `c2ws` (camera-to-world) must be accurate. Affine correction can fix per-camera scale and
      additive bias but cannot correct pose errors.
    - The bias `b` has units of length (meters) after unit normalization.
    - The method is sensitive to overlap between cameras; use object masks and multiple frames to
      improve robustness. Consider using the `--nframes` and `--random_frames` options to sample
      more diverse viewpoints.
    - The loss is not robustified here; for datasets with outliers, consider adding a robust loss
      (e.g., Huber) or rejecting far correspondences.
    """
    n = len(pcds)
    inv_c2ws = [np.linalg.inv(c2w) for c2w in c2ws]

    # Sample points from each pcd
    samples = []
    for pcd in pcds:
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            samples.append(np.zeros((0,3)))
            continue
        if pts.shape[0] > sample_per_pcd:
            idx = np.random.choice(pts.shape[0], sample_per_pcd, replace=False)
            pts_s = pts[idx]
        else:
            pts_s = pts
        samples.append(pts_s)

    def apply_affine(params):
        # params: [s0,b0,s1,b1,...]
        affined = []
        for i in range(n):
            s = params[2*i]
            b = params[2*i + 1]
            pts_world = samples[i]
            if pts_world.shape[0] == 0:
                affined.append(pts_world)
                continue
            # transform to camera coords
            pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1).T
            pts_cam = (inv_c2ws[i] @ pts_h).T[:, :3]
            z = pts_cam[:, 2]
            valid = z > 1e-6
            pts_cam_new = pts_cam.copy()
            if np.any(valid):
                z_valid = z[valid]
                z_new = s * z_valid + b
                f = z_new / z_valid
                pts_cam_new[valid, :] = pts_cam[valid, :] * f[:, None]
            # back to world
            pts_cam_new_h = np.concatenate([pts_cam_new, np.ones((pts_cam_new.shape[0], 1))], axis=1).T
            pts_world_new = (c2ws[i] @ pts_cam_new_h).T[:, :3]
            affined.append(pts_world_new)
        return affined

    def loss_fn(x):
        scaled = apply_affine(x)
        total = 0.0
        for i in range(n):
            src = scaled[i]
            if src.shape[0] == 0:
                continue
            others = np.vstack([scaled[j] for j in range(n) if j != i and scaled[j].shape[0] > 0]) if any(j != i and scaled[j].shape[0] > 0 for j in range(n)) else np.empty((0,3))
            if others.shape[0] == 0:
                continue
            tree = cKDTree(others)
            dists, _ = tree.query(src, k=1)
            total += dists.mean()
        return total

    x0 = np.hstack([np.array([1.0, 0.0]) for _ in range(n)])
    bnds = []
    for _ in range(n):
        bnds.append((bounds_s[0], bounds_s[1]))
        bnds.append((bounds_b[0], bounds_b[1]))
    res = minimize(loss_fn, x0, bounds=bnds, method='L-BFGS-B', options={'maxiter':200})
    params_opt = res.x

    # Apply to full point clouds
    affined_pcds = []
    for i in range(n):
        s = params_opt[2*i]
        b = params_opt[2*i + 1]
        pts_world = np.asarray(pcds[i].points)
        if pts_world.shape[0] == 0:
            affined_pcds.append(pcds[i])
            continue
        pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1).T
        pts_cam = (inv_c2ws[i] @ pts_h).T[:, :3]
        z = pts_cam[:, 2]
        valid = z > 1e-6
        pts_cam_new = pts_cam.copy()
        if np.any(valid):
            z_valid = z[valid]
            z_new = s * z_valid + b
            f = z_new / z_valid
            pts_cam_new[valid, :] = pts_cam[valid, :] * f[:, None]
        pts_cam_new_h = np.concatenate([pts_cam_new, np.ones((pts_cam_new.shape[0], 1))], axis=1).T
        pts_world_new = (c2ws[i] @ pts_cam_new_h).T[:, :3]
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(pts_world_new.astype(np.float32))
        if pcds[i].has_colors():
            pcd_new.colors = pcds[i].colors
        affined_pcds.append(pcd_new)

    # Return params as pairs per camera and the transformed pcds
    pairs = params_opt.reshape((-1, 2))
    return pairs, affined_pcds


def apply_affine_to_depth_file(src_path, s, b, out_path):
    """Load a depth .npy, apply z' = s*z + b (in meters), preserve original units, and save."""
    d = np.load(src_path)
    if not isinstance(d, np.ndarray):
        np.save(out_path, d)
        return
    # Detect units: heuristic same as load_depth_and_project
    is_mm = False
    try:
        if np.nanmax(d) > 100:
            is_mm = True
    except Exception:
        pass
    # Work in meters
    if is_mm:
        d_m = d.astype(np.float64) / 1000.0
    else:
        d_m = d.astype(np.float64)

    # Only apply to positive depths
    mask = d_m > 0
    d_new = d_m.copy()
    if np.any(mask):
        d_new[mask] = s * d_m[mask] + b
        # clip negatives to zero
        d_new = np.where(d_new > 0, d_new, 0.0)

    # Convert back to original units and dtype
    if is_mm:
        out = (d_new * 1000.0).astype(d.dtype)
    else:
        out = d_new.astype(d.dtype)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, out)


def save_scaled_depths(params, inputs, case_dir, correction='affine'):
    """Apply per-camera params to all depth .npy files and save under scaled_depth/<case_name>/<cam_idx>/.

    params: if correction=='affine' -> array-like of shape (N,2) with [s,b]
            if correction=='scale'  -> array-like of shape (N,) with s
    inputs: list of input folders per camera (in same order as params)
    """
    case_name = os.path.basename(os.path.normpath(case_dir))
    out_root = os.path.join(os.getcwd(), 'scaled_depth', case_name)
    os.makedirs(out_root, exist_ok=True)
    # Normalize params to numpy array and infer correction mode from shape when possible
    params_arr = np.array(params)
    mode = None
    if params_arr.ndim == 2 and params_arr.shape[1] == 2:
        mode = 'affine'
    elif params_arr.ndim == 1 or (params_arr.ndim == 2 and params_arr.shape[1] == 1):
        mode = 'scale'
        if params_arr.ndim == 2:
            params_arr = params_arr.flatten()
    else:
        # Fallback to provided 'correction' argument
        mode = 'affine' if correction == 'affine' else 'scale'

    for cam_idx, inf in enumerate(inputs):
        cam_out = os.path.join(out_root, str(cam_idx))
        os.makedirs(cam_out, exist_ok=True)
        files = sorted([f for f in os.listdir(inf) if f.endswith('.npy')])
        try:
            if mode == 'affine':
                s = float(params_arr[cam_idx][0])
                b = float(params_arr[cam_idx][1])
            else:
                s = float(params_arr[cam_idx])
                b = 0.0
        except Exception:
            # Defensive fallback: attempt to unpack or coerce
            try:
                maybe = params[cam_idx]
                if hasattr(maybe, '__len__') and len(maybe) >= 2:
                    s = float(maybe[0]); b = float(maybe[1])
                    mode = 'affine'
                else:
                    s = float(maybe); b = 0.0; mode = 'scale'
            except Exception as e:
                print(f'Could not interpret params for camera {cam_idx}: {e}; skipping')
                continue

        print(f'Applying correction to camera {cam_idx}: s={s}, b={b}, files={len(files)} -> {cam_out}')
        for fn in files:
            src = os.path.join(inf, fn)
            dst = os.path.join(cam_out, fn)
            try:
                apply_affine_to_depth_file(src, s, b, dst)
            except Exception as e:
                print(f'Failed to process {src}: {e}')
    print(f'Saved scaled depths to {out_root}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', type=str, default=None, help='Folder of depth .npy for camera 1')
    parser.add_argument('--input2', type=str, default=None, help='Folder of depth .npy for camera 2')
    parser.add_argument('--input3', type=str, default=None, help='Folder of depth .npy for camera 3')
    parser.add_argument('--calib', type=str, default=None, help='Path to calibrate.pkl')
    parser.add_argument('--meta', type=str, default=None, help='Path to metadata.json')
    parser.add_argument('--scale1', type=float, default=0.001, help='Depth scale for camera 1 (default: 0.001 for mm to m)')
    parser.add_argument('--scale2', type=float, default=0.001, help='Depth scale for camera 2 (default: 0.001 for mm to m)')
    parser.add_argument('--scale3', type=float, default=0.001, help='Depth scale for camera 3 (default: 0.001 for mm to m)')
    parser.add_argument('--mesh', action='store_true', help='Render point clouds as meshes for shaded visualization')
    parser.add_argument('--optimize_scales', action='store_true', help='Optimize per-camera uniform depth scales before alignment')
    parser.add_argument('--correction', type=str, default='scale', help='Type of depth correction: "scale" or "affine" (scale+bias)')
    parser.add_argument('--method', type=str, default='robust', help='Alignment method: "robust" (pose-init ICP with scale), "scale" (optimize scales), or "icp_refine" (point-to-plane ICP refinement)')
    parser.add_argument('--nframes', type=int, default=1, help='Number of frames per camera to use for scale optimization')
    parser.add_argument('--random_frames', action='store_true', help='Randomly sample frames (instead of taking first N)')
    parser.add_argument('--remove_plane', action='store_true', help='Remove dominant planar surface (e.g., table) before optimization')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory containing processed_masks.pkl or per-frame masks')
    parser.add_argument('--no_masks', action='store_true', help='Disable using processed masks even if available')
    parser.add_argument('--visualize_masks', action='store_true', help='Save per-frame mask overlays and show 3D masked points for debugging')
    parser.add_argument('--case_dir', type=str, default=None, help='Path to a dataset case folder under data/different_types/ to auto-load files')
    parser.add_argument('--plane_distance', type=float, default=0.01, help='RANSAC plane distance threshold (meters)')
    parser.add_argument('--plane_ransac_n', type=int, default=3, help='RANSAC plane sample size')
    parser.add_argument('--plane_iters', type=int, default=1000, help='RANSAC iterations for plane segmentation')
    args = parser.parse_args()

    # If a case directory is provided, auto-fill common arguments (only fill those not supplied)
    if args.case_dir is not None:
        base = args.case_dir
        # depth folders expected at <case>/depth/0,1,2
        if args.input1 is None:
            args.input1 = os.path.join(base, 'infer_depth', '0')
        if args.input2 is None:
            args.input2 = os.path.join(base, 'infer_depth', '1')
        if args.input3 is None:
            args.input3 = os.path.join(base, 'infer_depth', '2')
        # calibration and metadata
        cand_calib = os.path.join(base, 'calibrate.pkl')
        if args.calib is None and os.path.exists(cand_calib):
            args.calib = cand_calib
        cand_meta = os.path.join(base, 'metadata.json')
        if args.meta is None and os.path.exists(cand_meta):
            args.meta = cand_meta
        cand_mask = os.path.join(base, 'mask')
        if args.mask_dir is None and os.path.exists(cand_mask):
            args.mask_dir = cand_mask
        print(f'Auto-loaded case_dir: {base}')

    # Require either case_dir (which auto-fills) or explicit inputs
    if args.case_dir is None:
        missing = []
        for name in ('input1', 'input2', 'input3', 'calib', 'meta'):
            if getattr(args, name) is None:
                missing.append(name)
        if missing:
            parser.error(f"Either --case_dir must be provided, or these arguments are required: {', '.join(missing)}")

    # Load calibration
    with open(args.calib, 'rb') as f:
        c2ws = pickle.load(f)
    # Load intrinsics
    with open(args.meta, 'r') as f:
        meta = json.load(f)
    intrinsics = meta['intrinsics']

    # Collect N frames per camera (optionally random) and merge into one point cloud per camera
    files1 = sorted([f for f in os.listdir(args.input1) if f.endswith('.npy')])
    files2 = sorted([f for f in os.listdir(args.input2) if f.endswith('.npy')])
    files3 = sorted([f for f in os.listdir(args.input3) if f.endswith('.npy')])
    file_lists = [files1, files2, files3]
    inputs = [args.input1, args.input2, args.input3]
    merged_pcds = []
    # Load processed masks if provided and not disabled
    processed_masks = None
    if (not getattr(args, 'no_masks', False)) and args.mask_dir is not None:
        mask_pkl = os.path.join(args.mask_dir, 'processed_masks.pkl')
        if os.path.exists(mask_pkl):
            with open(mask_pkl, 'rb') as f:
                processed_masks = pickle.load(f)
            print(f'Loaded processed masks from {mask_pkl}')
        else:
            print(f'No processed_masks.pkl found at {mask_pkl}; proceeding without masks')
    for cam_idx, (folder, files, intr, c2w) in enumerate(zip(inputs, file_lists, intrinsics, c2ws)):
        if len(files) == 0:
            raise RuntimeError(f'No .npy files found in {folder}')
        nuse = min(args.nframes, len(files))
        if args.random_frames:
            chosen = list(np.random.choice(len(files), nuse, replace=False))
        else:
            chosen = list(range(nuse))
        pts_all = []
        for i in chosen:
            depth_path = os.path.join(folder, files[i])
            # determine actual frame id from filename (e.g., '10.npy' -> 10)
            try:
                frame_id = int(os.path.splitext(files[i])[0])
            except Exception:
                frame_id = None
            mask2d = None
            if (processed_masks is not None) and (frame_id is not None):
                # First try to union raw masks under mask_dir/<cam_idx>/*/<frame_id>.png
                union_mask = None
                try:
                    if args.mask_dir is not None:
                        pattern = os.path.join(args.mask_dir, str(cam_idx), '*', f'{frame_id}.png')
                        files_mask = sorted(glob.glob(pattern))
                        if len(files_mask) > 0:
                            for mpath in files_mask:
                                m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                                if m is None:
                                    continue
                                mb = (m > 0)
                                if union_mask is None:
                                    union_mask = mb
                                else:
                                    union_mask = np.logical_or(union_mask, mb)
                            if union_mask is not None:
                                mask2d = union_mask
                                try:
                                    s = int(np.sum(mask2d))
                                    print(f'Frame {frame_id} cam {cam_idx}: unioned raw masks count={s}, files={len(files_mask)}')
                                except Exception:
                                    print(f'Frame {frame_id} cam {cam_idx}: unioned raw masks present')
                except Exception as e:
                    print(f'Error reading raw masks for frame {frame_id} cam {cam_idx}: {e}')

                # If no raw union mask found, fall back to processed_masks.pkl
                if mask2d is None:
                    try:
                        frame_entry = processed_masks.get(frame_id, None) if isinstance(processed_masks, dict) else processed_masks[frame_id]
                        if frame_entry is None:
                            print(f'Frame {frame_id} not found in processed_masks')
                        else:
                            cam_entry = frame_entry.get(cam_idx, None) if isinstance(frame_entry, dict) else frame_entry[cam_idx]
                            if cam_entry is None:
                                print(f'No camera {cam_idx} entry in processed_masks for frame {frame_id}')
                            else:
                                mask2d = cam_entry.get('object', None) if isinstance(cam_entry, dict) else cam_entry['object']
                                if mask2d is not None:
                                    try:
                                        s = int(np.sum(mask2d))
                                        print(f'Frame {frame_id} cam {cam_idx}: mask present, sum={s}, shape={np.shape(mask2d)}')
                                    except Exception:
                                        print(f'Frame {frame_id} cam {cam_idx}: mask present (could not compute sum)')
                    except Exception as e:
                        print(f'Error accessing processed_masks for frame {frame_id} cam {cam_idx}: {e}')
                        mask2d = None

                # Save overlay image if requested and color frames available
                if (mask2d is not None) and getattr(args, 'visualize_masks', False) and getattr(args, 'case_dir', None) is not None:
                    try:
                        color_path = os.path.join(args.case_dir, 'color', str(cam_idx), f'{frame_id}.png')
                        if os.path.exists(color_path):
                            img = cv2.imread(color_path)
                            overlay = img.copy()
                            # color mask red overlay
                            overlay[mask2d.astype(bool)] = (0,0,255)
                            out = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                            debug_dir = os.path.join(args.case_dir, 'debug_masks')
                            os.makedirs(debug_dir, exist_ok=True)
                            out_path = os.path.join(debug_dir, f'cam{cam_idx}_frame{frame_id}.png')
                            cv2.imwrite(out_path, out)
                            print(f'Wrote mask overlay to {out_path}')
                        else:
                            # save raw mask
                            debug_dir = os.path.join(args.case_dir, 'debug_masks')
                            os.makedirs(debug_dir, exist_ok=True)
                            out_path = os.path.join(debug_dir, f'cam{cam_idx}_frame{frame_id}_mask.png')
                            cv2.imwrite(out_path, (mask2d.astype('uint8')*255))
                            print(f'Wrote raw mask to {out_path}')
                    except Exception as e:
                        print(f'Could not write mask overlay: {e}')
            elif (processed_masks is not None) and (frame_id is None):
                print(f'Could not parse frame id from filename {files[i]}; skipping mask lookup')
            pts = load_depth_and_project(depth_path, np.array(intr), np.array(c2w), mask2d=mask2d)
            pts_all.append(pts)
        # Concatenate per-frame points; handle empty arrays
        try:
            pts_cat = np.vstack(pts_all) if len(pts_all) > 0 else np.empty((0,3))
        except ValueError:
            # Could happen if pts_all contains incompatible shapes; fall back to concatenation with filtering
            pts_cat = np.vstack([p for p in pts_all if p is not None and p.size > 0]) if any(p is not None and p.size > 0 for p in pts_all) else np.empty((0,3))
        pcd = np_to_o3d(pts_cat)
        # Optionally remove dominant plane (e.g., table) that biases optimization
        if args.remove_plane:
            try:
                num_pts = len(pts_cat)
                print(f'Camera {cam_idx}: point count before plane removal: {num_pts}')
                if num_pts == 0:
                    print(f'Camera {cam_idx}: no points to segment; skipping plane removal')
                elif num_pts < args.plane_ransac_n:
                    print(f'Camera {cam_idx}: too few points ({num_pts}) for RANSAC (need {args.plane_ransac_n}); skipping plane removal')
                else:
                    plane_model, inliers = pcd.segment_plane(distance_threshold=args.plane_distance,
                                                             ransac_n=args.plane_ransac_n,
                                                             num_iterations=args.plane_iters)
                    print(f'Camera {cam_idx}: removed {len(inliers)} plane inliers')
                    mask = np.ones(len(pts_cat), dtype=bool)
                    mask[inliers] = False
                    pts_cat = pts_cat[mask]
                    pcd = np_to_o3d(pts_cat)
            except Exception as e:
                print(f'Plane segmentation failed for camera {cam_idx}: {e}')
        # color per camera
        color = [0,0,0]
        color[cam_idx%3] = 1
        pcd.paint_uniform_color(color)
        merged_pcds.append(pcd)

    pcds = merged_pcds

    visualize_point_clouds(pcds, 'Before Alignment', render_as_mesh=args.mesh)

    if args.optimize_scales:
        if args.correction == 'affine':
            print('Optimizing per-camera affine depth corrections (scale + bias)...')
            params_pairs, scaled_pcds = optimize_depth_affine(pcds, c2ws, intrinsics)
            print('Optimized (s,b) per camera:', params_pairs)
        else:
            print('Optimizing per-camera depth scales...')
            scales_opt, scaled_pcds = optimize_depth_scales(pcds, c2ws)
            # convert scales_opt (N,) to pairs-like structure for saving convenience
            params_pairs = np.array([[float(s), 0.0] for s in scales_opt])
            print('Optimized scales:', scales_opt)
        pcds = scaled_pcds
        visualize_point_clouds(pcds, 'After Scale Optimization', render_as_mesh=args.mesh)

        # Save scaled depth .npy files for all frames using the optimized affine/scale params
        try:
            inputs = [args.input1, args.input2, args.input3]
            save_scaled_depths(params_pairs, inputs, args.case_dir, correction=args.correction if args.correction is not None else ('affine' if params_pairs.shape[1]==2 else 'scale'))
        except Exception as e:
            print(f'Could not save scaled depths: {e}')

    # Choose alignment method
    if args.method == 'scale':
        aligned_pcds, transforms = align_point_clouds_with_scale(pcds)
    elif args.method == 'icp_refine':
        aligned_pcds, transforms = align_point_clouds_icp_refine(pcds, c2ws)
    else:
        aligned_pcds, transforms = align_point_clouds_with_scale_robust(pcds, c2ws)

    visualize_point_clouds(aligned_pcds, 'After Alignment (with scale)', render_as_mesh=args.mesh)

    # Optionally save the aligned clouds
    for i, pcd in enumerate(aligned_pcds):
        o3d.io.write_point_cloud(f'aligned_cam{i+1}.ply', pcd)
    print('Transformations:', transforms)
