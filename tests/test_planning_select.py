"""Marginal gains match direct log-determinants and greedy is consistent."""

import unittest

import numpy as np

from asid.fim.fim import posterior_std
from asid.fim.sketch import gain_against
from asid.planning.dictionary import Entry
from asid.planning.select import (
    accumulate,
    exact_ordered,
    greedy,
    shortlist,
    subspace_entry,
)


def entry(name: str, U: np.ndarray, gain: float = 0.0) -> Entry:
    return Entry(name=name, gain=gain, U=U)


class GainTests(unittest.TestCase):
    def test_woodbury_gain_matches_the_direct_logdet(self):
        rng = np.random.default_rng(1)
        first = rng.normal(size=(4, 3)) * 0.3
        second = rng.normal(size=(4, 3)) * 0.3
        information = np.eye(4) + first @ first.T
        direct = np.linalg.slogdet(
            np.eye(3) + second.T @ np.linalg.solve(information, second)
        )[1]
        self.assertAlmostEqual(gain_against(second, first), direct, places=8)
        self.assertAlmostEqual(
            gain_against(second, None),
            np.linalg.slogdet(np.eye(3) + second.T @ second)[1],
            places=8,
        )

    def test_greedy_orders_by_marginal_gain(self):
        basis = np.eye(4)[:, :3]
        entries = [
            entry("weak", 0.1 * basis),
            entry("strong", 0.7 * basis),
            entry("medium", 0.4 * basis),
        ]
        chosen = greedy(entries, 3)
        self.assertEqual(chosen[0][0].name, "strong")
        self.assertEqual([item[0].name for item in chosen][1:], ["medium", "weak"])
        gains = [item[1] for item in chosen]
        self.assertTrue(all(gain >= 0.0 for gain in gains))
        self.assertEqual([len(item[0].U) for item in chosen], [4, 4, 4])

        factors = None
        for chosen_entry, _ in chosen:
            factors = accumulate(factors, chosen_entry.U)
        total = np.linalg.slogdet(np.eye(4) + factors @ factors.T)[1]
        self.assertAlmostEqual(sum(gains), total, places=8)

    def test_greedy_stops_at_the_available_entries(self):
        entries = [entry("a", np.eye(2)), entry("b", np.eye(2))]
        self.assertEqual(len(greedy(entries, 5)), 2)
        self.assertEqual(greedy(entries, 0), [])


class ShortlistTests(unittest.TestCase):
    """The screen proposes candidates; the exact metric decides between them."""

    def setUp(self):
        self.entries = [
            entry("az270", np.eye(2), gain=233.23),
            entry("az090", np.eye(2), gain=230.75),
            entry("az000", np.eye(2), gain=219.48),
            entry("az180", np.eye(2), gain=209.25),
        ]

    def test_shortlist_takes_the_best_sketch_gains(self):
        self.assertEqual(
            [item.name for item in shortlist(self.entries, 2)], ["az270", "az090"]
        )
        self.assertEqual([item.name for item in shortlist(self.entries, 9)],
                         ["az270", "az090", "az000", "az180"])
        with self.assertRaises(ValueError):
            shortlist(self.entries, 0)

    def test_exact_order_overrides_the_sketch(self):
        # Measured at 124 frames / g = 20: the sketch puts az270 first by 2.5
        # nats while the exact Fisher puts az090 first by 64.3.
        gains = {"az270": 628.373, "az090": 692.671}
        ordered = exact_ordered(shortlist(self.entries, 2), gains)
        self.assertEqual([item.name for item in ordered], ["az090", "az270"])
        self.assertNotEqual(ordered[0].name, shortlist(self.entries, 1)[0].name)

    def test_exact_order_requires_every_shortlisted_gain(self):
        with self.assertRaises(ValueError) as caught:
            exact_ordered(shortlist(self.entries, 2), {"az270": 1.0})
        self.assertIn("az090", str(caught.exception))


class SubspaceTests(unittest.TestCase):
    def setUp(self):
        self.P = 6
        rng = np.random.default_rng(5)
        self.A = rng.normal(size=(self.P, 8)) * 0.4
        self.whitened = self.A @ self.A.T

    def test_full_rank_subspace_reproduces_the_fisher(self):
        entry = subspace_entry("full", self.whitened, np.eye(self.P))
        np.testing.assert_allclose(entry.U @ entry.U.T, self.whitened, atol=1e-10)
        self.assertAlmostEqual(
            entry.gain, float(np.linalg.slogdet(np.eye(self.P) + self.whitened)[1]), places=10
        )

    def test_shared_subspace_caps_the_union(self):
        Q = np.linalg.qr(np.random.default_rng(6).normal(size=(self.P, 2)))[0]
        first = subspace_entry("a", self.whitened, Q)
        other = np.linalg.qr(np.random.default_rng(7).normal(size=(self.P, self.P)))[0]
        second = subspace_entry("b", other @ self.whitened @ other.T, Q)
        union = np.concatenate([first.U, second.U], axis=1)
        self.assertLessEqual(np.linalg.matrix_rank(union), 2)
        full = float(np.linalg.slogdet(np.eye(self.P) + self.whitened)[1])
        self.assertLessEqual(first.gain, full + 1e-9)


class PosteriorTests(unittest.TestCase):
    def test_no_factors_gives_the_prior(self):
        prior = np.array([0.5, 0.3])
        np.testing.assert_allclose(posterior_std(prior, np.eye(2)), prior)

    def test_posterior_matches_the_direct_inverse(self):
        rng = np.random.default_rng(3)
        prior = np.array([0.5, 0.5, 0.3])
        factors = rng.normal(size=(3, 4)) * 0.4
        precision = np.eye(3) + factors @ factors.T
        direct = np.sqrt(np.diag(np.linalg.inv(precision)))
        np.testing.assert_allclose(posterior_std(prior, precision), prior * direct, atol=1e-10)
        self.assertTrue((posterior_std(prior, precision) <= prior).all())


if __name__ == "__main__":
    unittest.main()
