"""The dictionary recovers the whitened Fisher of a linear model at full rank."""

import unittest

import numpy as np

from asid.planning.dictionary import build_dictionary


class DictionaryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.P = 4
        self.A = rng.normal(size=(10, self.P))
        self.prior = np.array([0.5, 0.5, 0.3, 0.3])
        self.sigma = 0.005
        self.run = lambda phi: self.A @ phi
        self.H = (
            (self.A * self.prior[None, :] / self.sigma).T
            @ (self.A * self.prior[None, :] / self.sigma)
        )

    def test_full_rank_rank_one_factor_matches_the_fisher(self):
        entries = build_dictionary(
            [("a", self.run)],
            np.zeros(self.P),
            self.prior,
            self.sigma,
            rank=self.P,
            step=1e-2,
        )
        U = entries[0].U
        np.testing.assert_allclose(U @ U.T, self.H, atol=1e-7)
        expected = float(np.linalg.slogdet(np.eye(self.P) + self.H)[1])
        self.assertAlmostEqual(entries[0].gain, expected, places=6)

    def test_entries_are_ordered_and_share_the_subspace(self):
        scale = 3.0
        requests = [("weak", self.run), ("strong", lambda phi: scale * self.run(phi))]
        seen = []
        entries = build_dictionary(
            requests,
            np.zeros(self.P),
            self.prior,
            self.sigma,
            rank=2,
            step=1e-2,
            progress=lambda done, total: seen.append((done, total)),
        )
        self.assertEqual([entry.name for entry in entries], ["weak", "strong"])
        self.assertEqual(seen, [(1, 2), (2, 2)])
        self.assertEqual(entries[0].U.shape, (self.P, 2))
        self.assertGreater(entries[1].gain, entries[0].gain)
        self.assertLess(np.linalg.norm(entries[1].JQ - scale * entries[0].JQ), 1e-8)

    def test_independent_subspaces_beat_the_shared_rank_cap(self):
        requests = [(f"a{index}", self.run) for index in range(4)]
        shared = build_dictionary(
            requests, np.zeros(self.P), self.prior, self.sigma, rank=2, step=1e-2
        )
        independent = build_dictionary(
            requests,
            np.zeros(self.P),
            self.prior,
            self.sigma,
            rank=2,
            step=1e-2,
            subspace="independent",
        )
        shared_rank = np.linalg.matrix_rank(np.concatenate([e.U for e in shared], axis=1))
        independent_rank = np.linalg.matrix_rank(
            np.concatenate([e.U for e in independent], axis=1)
        )
        self.assertEqual(shared_rank, 2)
        self.assertEqual(independent_rank, self.P)

    def test_unknown_subspace_is_rejected(self):
        with self.assertRaises(ValueError):
            build_dictionary(
                [("a", self.run)],
                np.zeros(self.P),
                self.prior,
                self.sigma,
                rank=2,
                step=1e-2,
                subspace="per-action",
            )


if __name__ == "__main__":
    unittest.main()
