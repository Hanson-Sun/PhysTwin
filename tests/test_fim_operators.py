"""Observation selection and finite-difference operators."""

import unittest

import numpy as np
import torch

from asid.fim.fim import Fisher
from asid.fim.operators import (
    Observation,
    default_observation,
    directional_derivatives,
    jacobian,
)


class ObservationTests(unittest.TestCase):
    def test_selected_frames_are_flattened(self):
        trajectory = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        observation = Observation(frames=np.array([0, 2]), sigma=0.01)
        expected = trajectory[[0, 2]].numpy().reshape(-1).astype(np.float64)
        np.testing.assert_array_equal(observation.vector(trajectory), expected)

    def test_default_stride(self):
        observation = default_observation(60, stride=2, sigma=0.005)
        np.testing.assert_array_equal(observation.frames, np.arange(0, 60, 2))
        self.assertEqual(observation.sigma, 0.005)

    def test_mean_uses_noise_of_the_averaged_observation(self):
        frames = np.arange(4)
        sigma_frame = 0.2
        observation = Observation(frames, sigma_frame / np.sqrt(len(frames)), "mean")
        sensitivity = torch.tensor([[[1.0, 2.0, 3.0]]] * len(frames))
        jac = observation.vector(sensitivity)[None, :]
        fisher = Fisher.from_jacobian(jac, observation.sigma, np.ones(1))
        expected = len(frames) * np.square([1.0, 2.0, 3.0]).sum() / sigma_frame**2
        np.testing.assert_allclose(fisher.matrix, [[expected]])


class JacobianTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.dimension = 5
        self.output = 7
        self.matrix = rng.normal(size=(self.output, self.dimension))
        self.phi = rng.normal(size=self.dimension)

    def evaluate(self, phi):
        return self.matrix @ phi

    def test_one_sided_is_exact_for_linear(self):
        found = jacobian(self.evaluate, self.phi, step=1e-5, central=False)
        np.testing.assert_allclose(found, self.matrix.T, atol=1e-8)
        self.assertEqual(found.shape, (self.dimension, self.output))

    def test_central_is_exact_for_linear(self):
        found = jacobian(self.evaluate, self.phi, step=1e-3, central=True)
        np.testing.assert_allclose(found, self.matrix.T, atol=1e-8)

    def test_subset_of_columns(self):
        found = jacobian(self.evaluate, self.phi, step=1e-5, indices=[1, 3])
        np.testing.assert_allclose(found, self.matrix.T[[1, 3]], atol=1e-8)

    def test_directional_matches_matrix_product(self):
        directions = np.linalg.qr(
            np.random.default_rng(1).normal(size=(self.dimension, 3))
        )[0]
        found = directional_derivatives(self.evaluate, self.phi, directions, step=1e-5)
        np.testing.assert_allclose(found, self.matrix @ directions, atol=1e-8)
        self.assertEqual(found.shape, (self.output, 3))

    def test_central_handles_quadratic_exactly(self):
        weights = np.random.default_rng(2).normal(size=(self.dimension, self.output))

        def quadratic(phi):
            return phi @ weights + 0.5 * (phi**2) @ weights

        found = jacobian(quadratic, self.phi, step=1e-3, central=True)
        expected = weights * (1.0 + self.phi[:, None])
        np.testing.assert_allclose(found, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
