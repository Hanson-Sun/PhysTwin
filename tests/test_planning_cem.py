"""CEM on a toy quadratic: convergence, constraints, determinism."""

import unittest

import numpy as np

from asid.planning.cem import cem

TARGET = np.array([0.012, -0.008, 0.004])


def quadratic(x: np.ndarray) -> float:
    return -float(np.sum((np.asarray(x) - TARGET) ** 2))


class CemTest(unittest.TestCase):
    def test_converges_from_a_flat_start(self) -> None:
        lower, upper = np.full(3, -0.05), np.full(3, 0.05)
        result = cem(
            quadratic, np.zeros(3), lower, upper,
            samples=16, iterations=10, elites=3, seed=0,
        )
        self.assertGreater(result.score, quadratic(np.zeros(3)))
        self.assertLess(np.linalg.norm(result.vector - TARGET), 5e-3)
        self.assertGreater(result.score, -1e-3)
        self.assertEqual(result.evaluations, 1 + 10 * 16)
        self.assertEqual(result.history, sorted(result.history))
        self.assertEqual(len(result.history), 11)

    def test_infeasible_samples_are_dropped(self) -> None:
        lower, upper = np.full(3, -0.05), np.full(3, 0.05)

        def constrained(x: np.ndarray) -> float | None:
            return None if np.any(np.asarray(x) > 0.02) else quadratic(x)

        result = cem(
            constrained, np.zeros(3), lower, upper,
            samples=12, iterations=5, elites=2, seed=1,
        )
        self.assertLessEqual(float(np.max(result.vector)), 0.02 + 1e-12)
        self.assertGreaterEqual(result.score, quadratic(np.zeros(3)))

    def test_seed_reproduces(self) -> None:
        lower, upper = np.full(2, -1.0), np.full(2, 1.0)
        runs = [
            cem(lambda x: -float(np.sum(x ** 2)), np.ones(2), lower, upper,
                samples=4, iterations=3, seed=7)
            for _ in range(2)
        ]
        np.testing.assert_array_equal(runs[0].vector, runs[1].vector)
        self.assertEqual(runs[0].score, runs[1].score)

    def test_project_keeps_the_feasible_set(self) -> None:
        lower, upper = np.zeros(3), np.ones(3)
        result = cem(
            lambda v: -float(np.sum((v - np.array([0.2, 0.5, 0.8])) ** 2)),
            np.array([0.3, 0.4, 0.5]), lower, upper,
            samples=6, iterations=3, project=np.sort, seed=0,
        )
        self.assertTrue(np.all(np.diff(result.vector) >= 0.0))

    def test_pinned_coordinate_stays_put(self) -> None:
        lower = np.array([0.0, -0.05])
        upper = np.array([0.0, 0.05])
        result = cem(
            lambda x: -float((x[1] - 0.02) ** 2),
            np.zeros(2), lower, upper, samples=6, iterations=4, seed=0,
        )
        self.assertEqual(result.vector[0], 0.0)
        self.assertAlmostEqual(result.vector[1], 0.02, places=2)

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            cem(quadratic, np.zeros(3), np.ones(3), -np.ones(3))
        with self.assertRaises(ValueError):
            cem(quadratic, np.zeros(3), np.zeros(3), np.ones(3), samples=0)
        with self.assertRaises(ValueError):
            cem(lambda x: None, np.zeros(2), np.zeros(2), np.ones(2))


if __name__ == "__main__":
    unittest.main()
