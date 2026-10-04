"""Tests for depth-aware controller point filtering."""

import unittest

import numpy as np

from qqtt.utils.controller_collider import controller_depth_consistency


def make_camera(height=4, width=4):
    intrinsic = np.array([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]])
    w2c = np.eye(4)
    mask = np.zeros((height, width), dtype=bool)
    mask[1:3, 1:3] = True
    depth = np.ones((height, width), dtype=np.float32)
    masks = [{0: {"controller": mask}}]
    return intrinsic[None], w2c[None], masks, [depth]


def check(points, intrinsics, w2cs, masks, depths, **kwargs):
    return controller_depth_consistency(
        points,
        masks,
        intrinsics,
        w2cs,
        lambda frame, camera: depths[frame][camera],
        **kwargs,
    )


class ControllerDepthConsistencyTests(unittest.TestCase):
    def test_depth_matched_controller_pixel_supports_track(self):
        intrinsics, w2cs, masks, depths = make_camera()
        points = np.array([[[0.0, 0.0, 1.0]]])

        keep, support, conflict = check(
            points, intrinsics, w2cs, masks, [[depths[0]]]
        )

        self.assertTrue(keep[0])
        self.assertEqual(support[0], 1.0)
        self.assertEqual(conflict[0], 0.0)

    def test_point_behind_measured_surface_is_treated_as_occluded(self):
        intrinsics, w2cs, masks, depth = make_camera()
        masks[0][0]["controller"][:] = False
        points = np.array([[[0.0, 0.0, 1.5]]])

        keep, support, conflict = check(
            points,
            intrinsics,
            w2cs,
            masks,
            [[depth[0]]],
            min_support_fraction=0.0,
        )

        self.assertEqual(support[0], 0.0)
        self.assertEqual(conflict[0], 0.0)
        self.assertTrue(keep[0])

    def test_negative_fractional_pixel_is_outside_image(self):
        intrinsics = np.eye(3)[None]
        w2cs = np.eye(4)[None]
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        masks = [{0: {"controller": mask}}]
        depth = np.ones((2, 2), dtype=np.float32)
        points = np.array([[[-0.9, 0.5, 1.0]]])

        keep, support, conflict = check(
            points, intrinsics, w2cs, masks, [[depth]]
        )

        self.assertFalse(keep[0])
        self.assertEqual(support[0], 0.0)
        self.assertEqual(conflict[0], 0.0)

    def test_point_in_front_of_measured_depth_is_conflict(self):
        intrinsics, w2cs, masks, depth = make_camera()
        points = np.array([[[0.0, 0.0, 0.9]]])

        keep, support, conflict = check(points, intrinsics, w2cs, masks, [[depth[0]]])

        self.assertFalse(keep[0])
        self.assertEqual(support[0], 0.0)
        self.assertEqual(conflict[0], 1.0)

    def test_depth_matched_non_controller_pixel_is_conflict(self):
        intrinsics, w2cs, masks, depth = make_camera()
        masks[0][0]["controller"][:] = False
        points = np.array([[[0.0, 0.0, 1.0]]])

        keep, support, conflict = check(points, intrinsics, w2cs, masks, [[depth[0]]])

        self.assertFalse(keep[0])
        self.assertEqual(conflict[0], 1.0)

    def test_any_camera_can_provide_direct_support(self):
        intrinsic, w2c, _, depth = make_camera()
        empty = np.zeros_like(depth[0], dtype=bool)
        no_depth = np.zeros_like(depth[0])
        controller = np.zeros_like(empty)
        controller[1:3, 1:3] = True
        masks = [{0: {"controller": empty}, 1: {"controller": controller}}]
        points = np.array([[[0.0, 0.0, 1.0]]])

        keep, support, conflict = check(
            points,
            np.concatenate([intrinsic, intrinsic]),
            np.concatenate([w2c, w2c]),
            masks,
            [[no_depth, depth[0]]],
        )

        self.assertTrue(keep[0])
        self.assertEqual(support[0], 1.0)
        self.assertEqual(conflict[0], 0.0)

    def test_tracks_must_have_enough_direct_support(self):
        intrinsic, w2c, _, depth = make_camera()
        masks = [
            {0: {"controller": np.zeros_like(depth[0], dtype=bool)}}
            for _ in range(2)
        ]
        masks[0][0]["controller"][1:3, 1:3] = True
        points = np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]])

        keep, support, _ = check(
            points,
            intrinsic,
            w2c,
            masks,
            [[depth[0]], [depth[0]]],
            min_support_fraction=0.75,
        )

        self.assertFalse(keep[0])
        self.assertEqual(support[0], 0.5)

    def test_repeated_conflicts_drop_whole_track(self):
        intrinsic, w2c, _, depth = make_camera()
        mask = np.zeros_like(depth[0], dtype=bool)
        mask[1:3, 1:3] = True
        masks = [{0: {"controller": mask.copy()}} for _ in range(4)]
        points = np.repeat(np.array([[[0.0, 0.0, 0.9]]]), 4, axis=0)

        keep, _, conflict = check(
            points,
            intrinsic,
            w2c,
            masks,
            [[depth[0]]] * 4,
            max_conflict_fraction=0.05,
        )

        self.assertFalse(keep[0])
        self.assertEqual(conflict[0], 1.0)

    def test_rejects_bad_shapes_and_depth_dimensions(self):
        intrinsic, w2c, masks, depth = make_camera()
        with self.assertRaises(ValueError):
            check(np.zeros((1, 3)), intrinsic, w2c, masks, [[depth[0]]])
        with self.assertRaises(ValueError):
            check(
                np.array([[[0.0, 0.0, 1.0]]]),
                intrinsic,
                w2c,
                masks,
                [[np.zeros((3, 3))]],
            )


if __name__ == "__main__":
    unittest.main()