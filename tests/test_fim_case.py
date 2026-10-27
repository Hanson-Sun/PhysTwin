"""The reduced case keeps unique springs, correct rest lengths, and tethers."""

import unittest
from pathlib import Path

import numpy as np

from asid.fim.case import (
    build_controller_tethers,
    build_object_springs,
    farthest_point_indices,
    load_case,
    object_midpoints,
)

CASE_DIR = Path("data/different_types/sim_soft_sloth_push")


class FarthestPointTests(unittest.TestCase):
    def test_is_deterministic_and_unique(self):
        points = np.random.default_rng(0).normal(size=(200, 3))
        first = farthest_point_indices(points, 50, seed=3)
        second = farthest_point_indices(points, 50, seed=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 50)
        self.assertTrue(np.all(first < len(points)))

    def test_returns_every_point_when_count_is_large(self):
        points = np.zeros((7, 3))
        np.testing.assert_array_equal(farthest_point_indices(points, 10), np.arange(7))

    def test_midpoints_come_from_the_object_springs_only(self):
        nodes = np.zeros((4, 3))
        nodes[1, 0] = 0.02
        nodes[2, 0] = 0.04
        springs = np.array([[0, 1], [1, 2], [2, 3]])
        np.testing.assert_allclose(object_midpoints(nodes, springs, 2)[:, 0], [0.01, 0.03])


class SpringTests(unittest.TestCase):
    def setUp(self):
        axis = np.arange(3) * 0.05
        self.points = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)

    def test_pairs_are_unique_and_rest_lengths_match(self):
        springs, rest = build_object_springs(self.points, radius=0.11, max_neighbours=30)
        self.assertGreater(len(springs), 0)
        self.assertTrue(np.all(springs[:, 0] != springs[:, 1]))
        self.assertTrue(np.all(springs[:, 0] < springs[:, 1]))
        self.assertEqual(len({tuple(pair) for pair in springs}), len(springs))
        expected = np.linalg.norm(
            self.points[springs[:, 0]] - self.points[springs[:, 1]], axis=1
        )
        np.testing.assert_allclose(rest, expected, rtol=1e-6)

    def test_neighbour_cap_limits_pairs(self):
        capped, _ = build_object_springs(self.points, radius=0.2, max_neighbours=3)
        full, _ = build_object_springs(self.points, radius=0.2, max_neighbours=30)
        self.assertLess(len(capped), len(full))

    def test_tethers_index_the_controller_after_the_object(self):
        controller = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        springs, rest = build_controller_tethers(
            self.points, controller, radius=0.08, max_neighbours=10, num_object_nodes=len(self.points)
        )
        self.assertGreater(len(springs), 0)
        self.assertTrue(np.all(springs[:, 0] >= len(self.points)))
        self.assertTrue(np.all(springs[:, 1] < len(self.points)))
        self.assertTrue(np.all(rest > 0.0))


@unittest.skipUnless((CASE_DIR / "final_data.pkl").is_file(), "case data not available")
class RealCaseTests(unittest.TestCase):
    def test_reduced_case_shapes(self):
        case = load_case(CASE_DIR, num_nodes=500, max_frames=8)
        self.assertEqual(case.nodes.shape, (500, 3))
        self.assertEqual(case.controller.shape[1], 30)
        self.assertEqual(case.frame_count, 8)
        self.assertGreater(case.num_object_springs, 0)
        self.assertGreaterEqual(len(case.springs), case.num_object_springs)
        self.assertEqual(len(case.springs), len(case.rest_lengths))
        self.assertTrue(np.isfinite(case.nodes).all())


if __name__ == "__main__":
    unittest.main()
