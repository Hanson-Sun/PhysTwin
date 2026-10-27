"""Conditioned gains equal Woodbury gains and steer selection away from known directions."""

import unittest

import numpy as np

from asid.fim.fim import posterior_std
from asid.fim.sketch import gain_against
from asid.planning.select import accumulate
from asid.planning.sequential import marginal_gain, precision, select


class GainTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.first = rng.normal(size=(6, 3)) * 0.5
        self.second = rng.normal(size=(6, 3)) * 0.5

    def test_matches_the_woodbury_gain(self):
        matrix = precision([self.first @ self.first.T], 6)
        self.assertAlmostEqual(
            marginal_gain(matrix, self.second @ self.second.T),
            gain_against(self.second, self.first),
            places=8,
        )

    def test_prior_precision_gives_the_standalone_gain(self):
        alone = np.linalg.slogdet(np.eye(3) + self.second.T @ self.second)[1]
        self.assertAlmostEqual(
            marginal_gain(np.eye(6), self.second @ self.second.T), alone, places=8
        )

    def test_known_directions_earn_less_in_the_next_round(self):
        aligned = self.first @ self.first.T
        rng = np.random.default_rng(4)
        other = rng.normal(size=(6, 3))
        orthogonal = other - self.first @ np.linalg.lstsq(self.first, other, rcond=None)[0]
        fresh = orthogonal @ orthogonal.T

        first_round = select(np.eye(6), {"aligned": aligned, "fresh": fresh})[1]
        after = precision([aligned], 6)
        second_round = select(after, {"aligned": aligned, "fresh": fresh})[1]

        # Repeating a measurement is sublinear, and a disjoint block now wins.
        self.assertLess(second_round["aligned"], first_round["aligned"])
        self.assertLess(second_round["aligned"], 0.5 * first_round["aligned"])
        self.assertGreater(second_round["fresh"], second_round["aligned"])

    def test_selection_returns_the_best_name_and_all_scores(self):
        strong = 4.0 * self.first @ self.first.T
        weak = 0.1 * self.second @ self.second.T
        best, scores = select(np.eye(6), {"weak": weak, "strong": strong})
        self.assertEqual(best, "strong")
        self.assertEqual(set(scores), {"weak", "strong"})
        self.assertGreater(scores["strong"], scores["weak"])


class PosteriorTests(unittest.TestCase):
    def test_information_shrinks_the_posterior(self):
        prior = np.array([0.5, 0.5, 0.3] * 2)
        rng = np.random.default_rng(2)
        factors = rng.normal(size=(6, 4))
        matrix = precision([factors @ factors.T], 6)
        std = posterior_std(prior, matrix)
        self.assertTrue((std <= prior + 1e-12).all())
        self.assertTrue((std < prior).any())

    def test_no_information_returns_the_prior(self):
        prior = np.array([0.5, 0.3])
        np.testing.assert_allclose(posterior_std(prior, np.eye(2)), prior, atol=1e-12)

    def test_accumulated_precision_is_the_sum(self):
        rng = np.random.default_rng(3)
        first = rng.normal(size=(4, 2))
        second = rng.normal(size=(4, 2))
        matrix = precision([first @ first.T, second @ second.T], 4)
        np.testing.assert_allclose(matrix, np.eye(4) + first @ first.T + second @ second.T)
        factors = accumulate(accumulate(None, first), second)
        np.testing.assert_allclose(matrix, np.eye(4) + factors @ factors.T)


if __name__ == "__main__":
    unittest.main()
