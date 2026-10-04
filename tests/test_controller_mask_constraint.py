"""Tests for rejecting controller points that drift outside the masks."""

import unittest

import numpy as np

from qqtt.utils.controller_collider import points_inside_any_mask


def make_camera(height=4, width=4, fx=10.0, cx=2.0, cy=2.0):
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    w2c = np.eye(4)
    mask = np.zeros((height, width), dtype=bool)
    mask[1:3, 1:3] = True  # centre 2x2 region
    return intrinsic, w2c, mask


class PointsInsideAnyMaskTests(unittest.TestCase):
    def test_point_projected_into_the_mask_is_inside(self):
        intrinsic, w2c, mask = make_camera()
        # z=1, x=0 -> u = 10*0/1 + 2 = 2, v = 2 -> inside the centre region.
        points = np.array([[[0.0, 0.0, 1.0]]])
        masks = [{0: {"controller": mask}}]

        inside = points_inside_any_mask(
            points, masks, intrinsic[None], w2c[None]
        )

        self.assertTrue(inside[0, 0])

    def test_point_projected_outside_the_mask_is_outside(self):
        intrinsic, w2c, mask = make_camera()
        # u = 10*1/1 + 2 = 12 -> beyond the 4px-wide image.
        points = np.array([[[1.0, 0.0, 1.0]]])
        masks = [{0: {"controller": mask}}]

        inside = points_inside_any_mask(
            points, masks, intrinsic[None], w2c[None]
        )

        self.assertFalse(inside[0, 0])

    def test_negative_fractional_pixel_is_outside_image(self):
        intrinsic = np.eye(3)
        w2c = np.eye(4)
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        points = np.array([[[-0.9, 0.5, 1.0]]])
        masks = [{0: {"controller": mask}}]

        inside = points_inside_any_mask(
            points, masks, intrinsic[None], w2c[None]
        )

        self.assertFalse(inside[0, 0])

    def test_point_behind_the_camera_is_outside(self):
        intrinsic, w2c, mask = make_camera()
        points = np.array([[[0.0, 0.0, -1.0]]])
        masks = [{0: {"controller": mask}}]

        inside = points_inside_any_mask(
            points, masks, intrinsic[None], w2c[None]
        )

        self.assertFalse(inside[0, 0])

    def test_a_second_camera_can_cover_what_the_first_misses(self):
        intrinsic, w2c, mask = make_camera()
        # Empty mask for camera 0, populated mask for camera 1.
        empty = np.zeros_like(mask)
        points = np.array([[[0.0, 0.0, 1.0]]])
        masks = [{0: {"controller": empty}, 1: {"controller": mask}}]

        inside = points_inside_any_mask(
            points, masks, np.stack([intrinsic, intrinsic]), np.stack([w2c, w2c])
        )

        self.assertTrue(inside[0, 0])

    def test_no_masks_marks_everything_inside(self):
        points = np.zeros((2, 3, 3))
        inside = points_inside_any_mask(points, [], None, None)
        self.assertTrue(inside.all())

    def test_non_finite_points_are_never_inside(self):
        intrinsic, w2c, mask = make_camera()
        points = np.array([[[np.nan, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        masks = [{0: {"controller": mask}}]

        inside = points_inside_any_mask(
            points, masks, intrinsic[None], w2c[None]
        )

        self.assertFalse(inside[0, 0])
        self.assertTrue(inside[0, 1])

    def test_rejects_wrong_shaped_points(self):
        with self.assertRaises(ValueError):
            points_inside_any_mask(np.zeros((2, 3)), [], None, None)


if __name__ == "__main__":
    unittest.main()