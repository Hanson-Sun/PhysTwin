"""The exact gain decides, the screen's own pick is recorded, payloads round-trip."""

import unittest

import numpy as np

from asid.planning.dictionary import Entry
from asid.planning.measure import Choice, Decision, arbitrate, exact_fisher


def entry(name: str, gain: float) -> Entry:
    return Entry(name=name, gain=gain, U=np.eye(3))


class ExactFisherTest(unittest.TestCase):
    def test_matches_a_direct_logdet(self) -> None:
        """One-dimensional run: the exact gain is the Fisher log-determinant."""

        def run(phi):
            return np.array([3.0 * phi[0], 2.0 * phi[0]])

        fisher = exact_fisher(run, np.zeros(1), 0.01, 0.5, np.array([0.5]))
        # J = [[3], [2]] / sigma = 0.5, whitened by prior std 0.5.
        expected = np.log1p(((3.0 / 0.5) ** 2 + (2.0 / 0.5) ** 2) * 0.25)
        self.assertAlmostEqual(fisher.logdet(), float(expected), places=8)
        self.assertEqual(fisher.matrix.shape, (1, 1))


class ArbitrateTest(unittest.TestCase):
    """At 124 frames / g = 20 the screen ranked az270 first; exact reverses it."""

    def setUp(self) -> None:
        self.entries = [
            entry("az270", 233.23),
            entry("az090", 230.75),
            entry("az000", 226.00),
        ]
        self.exact = {"az270": 628.37, "az090": 692.67, "az000": 634.80}

    def exact_of(self, name: str) -> float:
        return self.exact[name]

    def test_exact_metric_decides_and_the_screen_is_still_recorded(self) -> None:
        decision = arbitrate(self.entries, 1, 2, self.exact_of, rank=12)
        self.assertEqual([choice.name for choice in decision.chosen], ["az090"])
        self.assertAlmostEqual(decision.chosen[0].gain, 692.67)
        self.assertEqual(decision.screened, ["az270"])
        self.assertEqual(decision.sketch_top, "az270")
        self.assertEqual(decision.exact_top, "az090")
        self.assertFalse(decision.agreement)
        self.assertEqual(set(decision.exact), {"az270", "az090"})

    def test_shortlist_of_one_measures_only_the_screen_best(self) -> None:
        seen = []

        def tracked(name: str) -> float:
            seen.append(name)
            return self.exact[name]

        decision = arbitrate(self.entries, 1, 1, tracked, rank=12)
        self.assertEqual(seen, ["az270"])
        self.assertEqual([choice.name for choice in decision.chosen], ["az270"])
        self.assertTrue(decision.agreement)

    def test_zero_shortlist_keeps_the_screen(self) -> None:
        decision = arbitrate(self.entries, 1, 0, self.exact_of, rank=12)
        self.assertEqual([choice.name for choice in decision.chosen], ["az270"])
        self.assertEqual(decision.screened, ["az270"])
        self.assertEqual(decision.exact, {})
        self.assertIsNone(decision.exact_top)
        self.assertIn("rank-12", decision.basis)


class DecisionPayloadTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        decision = arbitrate(
            [entry("az270", 233.23), entry("az090", 230.75)],
            1,
            2,
            {"az270": 628.37, "az090": 692.67}.__getitem__,
            rank=12,
        )
        payload = decision.payload()
        self.assertEqual(payload["selected"], [{"name": "az090", "marginal_gain": 692.67}])
        self.assertEqual(payload["sketch_selected"], ["az270"])
        self.assertFalse(payload["rank_agreement"]["top1_match"])
        self.assertEqual(Decision.from_payload(payload), decision)

    def test_legacy_payload_without_the_screen_label(self) -> None:
        """Runs before exact arbitration recorded only ``selected``."""

        back = Decision.from_payload(
            {"selected": [{"name": "az090", "marginal_gain": 1.0}], "exact": {}}
        )
        self.assertEqual(back.screened, ["az090"])
        self.assertEqual([choice.name for choice in back.chosen], ["az090"])
        self.assertEqual(back.basis, "")
        self.assertIsNone(back.exact_top)
        self.assertFalse(back.agreement)


if __name__ == "__main__":
    unittest.main()
