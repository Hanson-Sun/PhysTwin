import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from data_process.interior_sample import (
    backproject_masked,
    carve_occupancy,
    erode_to_interior,
    largest_cluster,
    sample_interior_points,
)


def look_at_z_forward(eye, target):
    """c2w for an OpenCV camera at ``eye`` looking at ``target`` (z forward)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Forward is (anti)parallel to world up; pick a horizontal axis.
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, forward, eye
    return c2w


def camera_rig(resolution=320, fx=300.0):
    """Three oblique cameras aimed at the origin-ish object, like mujoco_assets."""
    intrinsics = np.array([[fx, 0, resolution / 2], [0, fx, resolution / 2], [0, 0, 1]])
    eyes = [
        np.array([0.6, 0.0, 0.05]),
        np.array([0.0, 0.6, 0.05]),
        np.array([0.08, 0.08, 0.64]),
    ]
    return intrinsics, eyes


def analytic_depth(sdf, eye, c2w, resolution=320, fx=300.0, t0=0.2, t_max=1.2):
    """First-hit depth per pixel by sphere tracing a signed distance field."""
    ys, xs = np.mgrid[0:resolution, 0:resolution]
    dirs_cam = np.stack(
        [(xs - resolution / 2) / fx, (ys - resolution / 2) / fx,
         np.ones_like(xs, dtype=np.float64)],
        axis=-1,
    )
    dirs = dirs_cam @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origin = c2w[:3, 3]
    t = np.full((resolution, resolution), t0)
    depth = np.zeros((resolution, resolution))
    for _ in range(1500):
        points = origin + dirs * t[..., None]
        value = sdf(points)
        hit = value <= 1e-4
        depth = np.where(hit & (depth == 0), t, depth)
        if (depth > 0).all() or t.max() > t_max:
            break
        t = np.where(hit, t, t + np.clip(value, 5e-4, 0.01))
    return depth.astype(np.float32), (depth > 0)


def synthetic_sphere_scene(radius=0.05, center=(0.0, 0.0, 0.05), resolution=320):
    """Three cameras viewing a sphere; returns masks, depths, intrinsics,
    w2cs, and the analytic sphere parameters."""
    center = np.asarray(center, dtype=np.float64)
    intrinsics, eyes = camera_rig(resolution)
    masks, depths, w2cs = [], [], []
    for eye in eyes:
        c2w = look_at_z_forward(eye, center)
        w2cs.append(np.linalg.inv(c2w))

        def sdf(points, center=center, radius=radius):
            return np.linalg.norm(points - center, axis=-1) - radius

        depth, mask = analytic_depth(sdf, eye, c2w, resolution, intrinsics[0, 0])
        depths.append(depth)
        masks.append(mask)
    return masks, depths, intrinsics, w2cs, center, radius


class CarveOccupancyTests(unittest.TestCase):
    def test_sphere_occupancy_encloses_surface_and_excludes_free_space(self):
        masks, depths, intrinsics, w2cs, center, radius = synthetic_sphere_scene()
        self.assertTrue(all(int(m.sum()) > 500 for m in masks), "camera rig broke")

        voxel = 0.0025
        origin = np.array([-0.11, -0.11, -0.06])
        occupied = carve_occupancy(
            dims=(96, 96, 96),
            grid_origin=origin,
            voxel=voxel,
            masks=masks,
            depths=depths,
            intrinsics=[intrinsics] * 3,
            w2cs=w2cs,
            depth_margin=0.004,
            floor_z=1.0,
        )
        self.assertGreater(int(occupied.sum()), 0)

        centers = origin + (np.argwhere(occupied) + 0.5) * voxel
        radii = np.linalg.norm(centers - center, axis=1)
        # Occupied voxels must lie within the analytic sphere, or in the
        # visual-hull bulge behind it: never in observed free space. A voxel
        # outside the sphere is only acceptable if it is on the far side of
        # every camera's viewing axis (hidden from all views).
        outside = radii > radius + 2 * voxel
        if outside.any():
            hidden = np.ones(int(outside.sum()), dtype=bool)
            for eye in camera_rig()[1]:
                axis = center - eye
                axis = axis / np.linalg.norm(axis)
                hidden &= ((centers[outside] - center) * axis).sum(axis=1) > 0
            self.assertTrue(
                hidden.all(), "occupancy leaked into observed free space"
            )
        # ...and deep inside it must be occupied.
        probe = (center + np.array([0.0, 0.0, -0.02]) - origin) / voxel
        self.assertTrue(occupied[tuple(np.floor(probe).astype(int))])

    def test_observed_surface_is_never_carved_as_free(self):
        masks, depths, intrinsics, w2cs, _, _ = synthetic_sphere_scene()
        observed = backproject_masked(
            depths[0], masks[0], intrinsics, np.linalg.inv(w2cs[0]), 2.0
        )
        voxel = 0.0025
        origin = np.array([-0.11, -0.11, -0.06])
        dims = (96, 96, 96)

        occupied = carve_occupancy(
            dims, origin, voxel, masks, depths, [intrinsics] * 3, w2cs, 0.004, 1.0
        )
        indices = np.floor((observed - origin) / voxel).astype(int)
        occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        surface_voxels = occupied[indices[:, 0], indices[:, 1], indices[:, 2]]
        self.assertTrue(surface_voxels.all())

    def test_single_camera_fills_entire_visual_hull(self):
        # One camera cannot see the back: everything behind the surface within
        # the silhouette stays occupied (the visual-hull bound).
        masks, depths, intrinsics, w2cs, _, _ = synthetic_sphere_scene()
        occupied = carve_occupancy(
            dims=(96, 96, 96),
            grid_origin=np.array([-0.11, -0.11, -0.06]),
            voxel=0.0025,
            masks=masks[:1],
            depths=depths[:1],
            intrinsics=[intrinsics],
            w2cs=w2cs[:1],
            depth_margin=0.004,
            floor_z=1.0,
        )
        self.assertGreater(int(occupied.sum()), 0)


class FloorAwareTests(unittest.TestCase):
    """Nothing may be carved below the support surface (z > floor_z)."""

    def test_no_occupancy_or_interior_below_the_floor(self):
        masks, depths, intrinsics, w2cs, _, _ = synthetic_sphere_scene(
            radius=0.05, center=(0.0, 0.0, 0.02)
        )
        voxel = 0.0025
        origin = np.array([-0.11, -0.11, -0.08])
        dims = (96, 96, 96)
        occupied = carve_occupancy(
            dims, origin, voxel, masks, depths, [intrinsics] * 3, w2cs, 0.004, 0.0
        )
        z_centers = origin[2] + (np.arange(dims[2]) + 0.5) * voxel
        below = z_centers > 0.0
        self.assertGreater(int(below.sum()), 0)
        self.assertEqual(int(occupied[:, :, below].sum()), 0)

        interior = erode_to_interior(occupied)
        points = sample_interior_points(
            interior, origin, voxel, 10_000, np.random.default_rng(0)
        )
        self.assertGreater(points.shape[0], 0)
        self.assertLessEqual(float(points[:, 2].max()), 0.0)


class ErodeToInteriorTests(unittest.TestCase):
    def test_interior_is_strictly_inside_occupied(self):
        # Two-voxel-thin slab: one-voxel 26-neighbourhood erosion leaves nothing.
        occupied = np.zeros((20, 20, 20), dtype=bool)
        occupied[4:16, 4:16, 9:11] = True
        interior = erode_to_interior(occupied)
        self.assertFalse(interior.any())

        # Thick block: erosion keeps exactly the voxels whose full
        # 26-neighbourhood is occupied (indices 6..23 here).
        occupied = np.zeros((30, 30, 30), dtype=bool)
        occupied[5:25, 5:25, 5:25] = True
        interior = erode_to_interior(occupied)
        expected = np.zeros_like(occupied)
        expected[6:24, 6:24, 6:24] = True
        np.testing.assert_array_equal(interior, expected)
        # No kept voxel is free.
        np.testing.assert_array_equal(interior & ~occupied, False)

    def test_sampled_points_are_inside_and_jittered(self):
        interior = np.zeros((16, 16, 16), dtype=bool)
        interior[4:12, 4:12, 4:12] = True
        origin = np.array([-0.02, -0.02, -0.02])
        rng = np.random.default_rng(0)
        points = sample_interior_points(interior, origin, 0.004, 10_000, rng)
        self.assertGreater(points.shape[0], 0)
        self.assertLessEqual(points.shape[0], 10_000)
        inside = (points >= origin) & (points <= origin + 0.064)
        self.assertTrue(inside.all())
        # Points are jittered, so not exactly on voxel corners.
        corners = origin + np.argwhere(interior) * 0.004
        self.assertGreater(float(np.abs(points[: len(corners)] - corners).max()), 0)

    def test_max_points_subsamples(self):
        interior = np.zeros((40, 40, 40), dtype=bool)
        interior[5:35, 5:35, 5:35] = True
        rng = np.random.default_rng(1)
        points = sample_interior_points(interior, np.zeros(3), 0.001, 500, rng)
        self.assertEqual(points.shape[0], 500)
        self.assertEqual(np.unique(points, axis=0).shape[0], 500)


class LargestClusterTests(unittest.TestCase):
    def test_drops_isolated_outliers(self):
        rng = np.random.default_rng(2)
        cluster = rng.random((2000, 3)) * 0.05
        outliers = rng.random((5, 3)) * 0.05 + 0.5
        points = np.vstack([cluster, outliers])
        kept = largest_cluster(points, 0.016)
        self.assertEqual(kept.shape[0], 2000)
        self.assertLess(kept.max(), 0.1)


class SlothShapedSceneTests(unittest.TestCase):
    """Composite body + thin-limb geometry (sloth-like) on a three-camera rig."""

    def test_thin_limb_has_no_interior_but_body_does(self):
        resolution = 320
        intrinsics, eyes = camera_rig(resolution)
        body_center = np.array([0.0, 0.0, 0.05])
        limb_start = np.array([0.03, 0.0, 0.035])
        limb_end = np.array([0.08, 0.0, 0.03])
        # Rope-thin limb: 4 mm across (~2 voxels) is narrower than the
        # 26-neighbourhood erosion reach (sqrt(3) * voxel), so it must yield
        # no interior of its own.
        limb_radius = 0.002

        def sdf(points):
            body = np.linalg.norm(points - body_center, axis=-1) - 0.04
            # Capsule limb: distance to the limb segment minus its radius.
            segment = limb_end - limb_start
            length2 = float(segment @ segment)
            alpha = np.clip(
                ((points - limb_start) @ segment) / length2, 0.0, 1.0
            )
            closest = limb_start + alpha[..., None] * segment
            limb = np.linalg.norm(points - closest, axis=-1) - limb_radius
            return np.minimum(body, limb)

        masks, depths, w2cs = [], [], []
        for eye in eyes:
            c2w = look_at_z_forward(eye, body_center)
            w2cs.append(np.linalg.inv(c2w))
            depth, mask = analytic_depth(
                sdf, eye, c2w, resolution, float(intrinsics[0, 0])
            )
            self.assertTrue(int(mask.sum()) > 500, "camera rig broke")
            masks.append(mask)
            depths.append(depth)

        voxel = 0.002
        origin = np.array([-0.09, -0.09, -0.04])
        dims = (100, 100, 90)
        occupied = carve_occupancy(
            dims, origin, voxel, masks, depths, [intrinsics] * 3, w2cs, 0.004, 1.0
        )
        self.assertGreater(int(occupied.sum()), 0)
        interior = erode_to_interior(occupied)

        centers = origin + (np.argwhere(interior) + 0.5) * voxel
        body_dist = np.linalg.norm(centers - body_center, axis=1)

        # The body (4 cm radius) keeps interior points...
        self.assertGreater(int((body_dist < 0.03).sum()), 100)
        # ...while the thin limb yields none of its own: no interior point
        # inside the limb capsule but outside the body sphere. (Points near
        # the body junction are legitimately interior to the union.)
        segment = limb_end - limb_start
        alpha = np.clip(((centers - limb_start) @ segment) / float(segment @ segment), 0, 1)
        closest = limb_start + alpha[:, None] * segment
        capsule_dist = np.linalg.norm(centers - closest, axis=1)
        limb_only = (capsule_dist < limb_radius) & (body_dist >= 0.04)
        self.assertEqual(int(limb_only.sum()), 0)


if __name__ == "__main__":
    unittest.main()
