"""Tests for filling occluded controller points across frames.

The gripper collider keeps points the grasped object hides. Those frames are
rebuilt from the neighbouring observations, so the fill must never invent a
position: it interpolates between two real measurements, or clamps to one.
"""

import unittest

import numpy as np

from qqtt.utils.controller_collider import (
    count_visible_controller_frames,
    fill_occluded_controller_points,
)


class FillOccludedControllerPointsTests(unittest.TestCase):
    def test_visible_frames_are_untouched(self):
        points = np.random.default_rng(0).normal(size=(6, 4, 3))
        visibilities = np.ones((6, 4), dtype=bool)
        filled, never = fill_occluded_controller_points(points, visibilities)
        np.testing.assert_allclose(filled, points)
        self.assertEqual(never, 0)

    def test_interpolates_between_bracketing_observations(self):
        # A point moving linearly in x, hidden for one frame in the middle.
        points = np.zeros((5, 1, 3))
        points[:, 0, 0] = [0.0, 1.0, 0.0, 3.0, 4.0]
        visibilities = np.ones((5, 1), dtype=bool)
        visibilities[2, 0] = False

        filled, never = fill_occluded_controller_points(points, visibilities)

        self.assertEqual(never, 0)
        # Frame 1 is 1.0 and frame 3 is 3.0, so the midpoint is 2.0. The
        # un-filled array held a zero here, which would place the collider
        # at the world origin.
        self.assertAlmostEqual(float(filled[2, 0, 0]), 2.0)
        self.assertEqual(int(np.count_nonzero(filled[2, 0, 1:])), 0)

    def test_interpolates_across_a_long_gap(self):
        points = np.zeros((6, 1, 3))
        points[:, 0, 0] = [0.0, 0.0, 0.0, 3.0, 6.0, 9.0]
        visibilities = np.ones((6, 1), dtype=bool)
        visibilities[1:3, 0] = False

        filled, _ = fill_occluded_controller_points(points, visibilities)

        self.assertAlmostEqual(float(filled[1, 0, 0]), 1.0)
        self.assertAlmostEqual(float(filled[2, 0, 0]), 2.0)

    def test_clamps_to_the_only_side_rather_than_extrapolating(self):
        # Visible only at the first frame: later frames must hold, not drift.
        points = np.zeros((4, 1, 3))
        points[:, 0, 0] = [2.0, 0.0, 0.0, 0.0]
        visibilities = np.zeros((4, 1), dtype=bool)
        visibilities[0, 0] = True

        filled, never = fill_occluded_controller_points(points, visibilities)

        self.assertEqual(never, 0)
        for frame in (1, 2, 3):
            self.assertAlmostEqual(float(filled[frame, 0, 0]), 2.0)

    def test_point_visible_in_no_frame_is_reported_and_nan(self):
        points = np.zeros((3, 2, 3))
        visibilities = np.zeros((3, 2), dtype=bool)
        visibilities[:, 0] = True

        filled, never = fill_occluded_controller_points(points, visibilities)

        self.assertEqual(never, 3)
        self.assertTrue(np.isnan(filled[:, 1, :]).all())
        self.assertFalse(np.isnan(filled[:, 0, :]).any())

    def test_does_not_modify_the_input(self):
        points = np.zeros((3, 1, 3))
        visibilities = np.ones((3, 1), dtype=bool)
        visibilities[1, 0] = False
        original = points.copy()
        fill_occluded_controller_points(points, visibilities)
        np.testing.assert_allclose(points, original)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            fill_occluded_controller_points(
                np.zeros((3, 2, 3)), np.ones((3, 3), dtype=bool)
            )


class DenseControllerSelectionTests(unittest.TestCase):
    """Threshold deciding which claw points enter the dense collider."""

    def test_keeps_points_at_or_above_the_threshold(self):
        visibilities = np.zeros((10, 3), dtype=bool)
        visibilities[:7, 0] = True
        visibilities[:6, 1] = True
        visibilities[:, 2] = True

        kept, min_visible = count_visible_controller_frames(visibilities, 0.7)

        self.assertEqual(min_visible, 7)
        np.testing.assert_array_equal(kept, [True, False, True])

    def test_threshold_of_one_matches_all_frames_intersection(self):
        rng = np.random.default_rng(3)
        visibilities = rng.random((12, 40)) < 0.5

        kept, _ = count_visible_controller_frames(visibilities, 1.0)

        np.testing.assert_array_equal(kept, visibilities.all(axis=0))

    def test_rejects_threshold_outside_range(self):
        with self.assertRaises(ValueError):
            count_visible_controller_frames(np.ones((4, 2), dtype=bool), 0.0)


if __name__ == "__main__":
    unittest.main()
