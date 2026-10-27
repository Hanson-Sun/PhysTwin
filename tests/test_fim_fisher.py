"""Exact Fisher and q=0 sketch."""

import unittest

import numpy as np

from asid.fim.fim import Fisher, Prior
from asid.fim.sketch import gain_against, randomized_sketch


class FisherTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.jac = rng.normal(size=(7, 20))
        self.sigma = 0.01
        self.prior = Prior().std(3)
        self.fisher = Fisher.from_jacobian(self.jac, self.sigma, self.prior)

    def test_matrix_is_symmetric_psd(self):
        np.testing.assert_allclose(self.fisher.matrix, self.fisher.matrix.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.fisher.matrix).min(), -1e-10)
        expected = self.jac @ self.jac.T / self.sigma**2
        np.testing.assert_allclose(self.fisher.matrix, expected)

    def test_whitened_spectrum_and_logdet(self):
        whitened = self.fisher.whitened
        expected = self.prior[:, None] * self.fisher.matrix * self.prior[None, :]
        np.testing.assert_allclose(whitened, expected)
        values = self.fisher.eigenvalues()
        self.assertEqual(values.shape, (7,))
        np.testing.assert_allclose(
            self.fisher.logdet(),
            np.linalg.slogdet(np.eye(7) + whitened)[1],
        )

    def test_posterior_std_shrinks(self):
        std = self.fisher.posterior_std()
        self.assertTrue(np.all(std > 0.0))
        self.assertTrue(np.all(std <= self.prior + 1e-12))

    def test_noise_and_independent_repeats_scale_information(self):
        baseline = self.fisher.matrix
        noisy = Fisher.from_jacobian(self.jac, 2 * self.sigma, self.prior)
        repeated = Fisher.from_jacobian(
            np.tile(self.jac, (1, 3)), self.sigma, self.prior
        )
        np.testing.assert_allclose(noisy.matrix, baseline / 4)
        np.testing.assert_allclose(repeated.matrix, 3 * baseline)

    def test_gaussian_score_matches_log_likelihood_gradient(self):
        rng = np.random.default_rng(3)
        phi = rng.normal(size=7)
        observation = rng.normal(size=20)
        jac = self.jac.T

        def log_likelihood(parameters):
            residual = observation - jac @ parameters
            return -0.5 * np.dot(residual, residual) / self.sigma**2

        score = jac.T @ (observation - jac @ phi) / self.sigma**2
        step = 1e-5
        numerical = np.array([
            (log_likelihood(phi + step * np.eye(7)[i])
             - log_likelihood(phi - step * np.eye(7)[i])) / (2 * step)
            for i in range(7)
        ])
        np.testing.assert_allclose(score, numerical, rtol=1e-8, atol=1e-5)

    def test_correlated_observations_require_whitening(self):
        jac = np.array([[1.0, 2.0], [3.0, -1.0]])
        covariance = np.array([[2.0, 0.8], [0.8, 1.0]])
        whitened_jac = np.linalg.solve(np.linalg.cholesky(covariance), jac)
        fisher = Fisher.from_jacobian(whitened_jac.T, 1.0, np.ones(2))
        expected = jac.T @ np.linalg.solve(covariance, jac)
        np.testing.assert_allclose(fisher.matrix, expected)
        self.assertFalse(np.allclose(fisher.matrix, jac.T @ jac))

    def test_grouped_information_is_the_pullback(self):
        rng = np.random.default_rng(2)
        full_jac = rng.normal(size=(20, 6))
        basis = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0]])
        grouped_jac = full_jac @ basis
        grouped = Fisher.from_jacobian(grouped_jac.T, self.sigma, np.ones(2))
        full = full_jac.T @ full_jac / self.sigma**2
        np.testing.assert_allclose(grouped.matrix, basis.T @ full @ basis)


class SketchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.P = 8
        self.D = 30
        singular = np.array([20.0, 10.0, 3.0, 1.0, 0.2, 0.05, 0.01, 0.001])
        left = np.linalg.qr(rng.normal(size=(self.D, self.P)))[0]
        self.jac = ((left * singular) @ rng.normal(size=(self.P, self.P))).T
        # run returns y = J^T phi, so the FD Jacobian is J.
        self.run = lambda phi: self.jac.T @ phi
        self.phi = rng.normal(size=self.P)
        self.prior = np.full(self.P, 0.5)
        self.sigma = 0.01

    def test_full_rank_reconstructs_the_fisher(self):
        sketch = randomized_sketch(
            self.run, self.phi, self.prior, self.sigma, rank=self.P, step=1e-5
        )
        Jw = self.jac / self.sigma * self.prior[:, None]
        H = Jw @ Jw.T
        np.testing.assert_allclose(sketch.U @ sketch.U.T, H, atol=1e-8)
        exact = np.linalg.slogdet(np.eye(self.P) + H)[1]
        np.testing.assert_allclose(sketch.gain, exact, atol=1e-8)

    def test_low_rank_underestimates_but_tracks(self):
        Jw = self.jac / self.sigma * self.prior[:, None]
        exact = np.linalg.slogdet(np.eye(self.P) + Jw @ Jw.T)[1]
        sketch = randomized_sketch(
            self.run, self.phi, self.prior, self.sigma, rank=4, step=1e-5, seed=3
        )
        self.assertLessEqual(sketch.gain, exact + 1e-8)
        self.assertGreater(sketch.gain, 0.5 * exact)

    def test_directions_are_orthonormal(self):
        sketch = randomized_sketch(
            self.run, self.phi, self.prior, self.sigma, rank=3, step=1e-5
        )
        scaled = sketch.directions / self.prior[:, None]
        np.testing.assert_allclose(scaled.T @ scaled, np.eye(3), atol=1e-10)

    def test_incremental_gain_matches_recomputation(self):
        rng = np.random.default_rng(4)
        current = rng.normal(size=(self.P, 3))
        U = rng.normal(size=(self.P, 2))
        H = np.eye(self.P) + current @ current.T
        expected = np.linalg.slogdet(H + U @ U.T)[1] - np.linalg.slogdet(H)[1]
        self.assertAlmostEqual(gain_against(U, current), expected, places=8)


if __name__ == "__main__":
    unittest.main()
