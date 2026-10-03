"""CMA-ES generation loop: parity with es.optimize() and stall-based early stop."""
import unittest

import cma
import numpy as np

from qqtt.engine.cma_optimize_warp import OptimizerCMA
from qqtt.utils import cfg


def make_strategy():
    return cma.CMAEvolutionStrategy(
        [0.5] * 11, 1 / 6, {"bounds": [0.0, 1.0], "seed": 42}
    )


def make_optimizer(objective):
    """OptimizerCMA without the dataset/CUDA setup in __init__."""
    optimizer = OptimizerCMA.__new__(OptimizerCMA)
    optimizer.objective = objective
    return optimizer


def quadratic(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum((x - 0.3) ** 2))


class CmaEarlyStopTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: getattr(cfg, name)
            for name in (
                "cma_early_stop_patience",
                "cma_early_stop_min_delta",
            )
        }

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(cfg, name, value)

    def test_disabled_matches_es_optimize(self):
        cfg.cma_early_stop_patience = 0
        optimizer = make_optimizer(quadratic)

        reference = make_strategy()
        reference.optimize(quadratic, iterations=12)

        candidate = make_strategy()
        optimizer._run_generations(candidate, 12)

        self.assertEqual(candidate.countiter, reference.countiter)
        self.assertEqual(candidate.countiter, 12)
        self.assertAlmostEqual(candidate.result[1], reference.result[1], places=12)
        np.testing.assert_allclose(candidate.result[0], reference.result[0])

    def test_stall_stops_early_and_keeps_best_candidate(self):
        # A delta of 1.0 can never be met, so every generation after the first
        # counts as stalled and the run must stop as soon as it is allowed.
        cfg.cma_early_stop_patience = 2
        cfg.cma_early_stop_min_delta = 1.0

        seen = []

        def counting(x):
            seen.append(x)
            return quadratic(x)

        optimizer = make_optimizer(counting)
        strategy = make_strategy()
        optimizer._run_generations(strategy, 20)

        # Generation 1 seeds the best; 2 and 3 both stall, so the patience of 2
        # is met at generation 3.
        self.assertEqual(strategy.countiter, 3)
        self.assertEqual(len(seen), 3 * strategy.popsize)

    def test_progress_resets_the_stall_counter(self):
        # The best candidate keeps improving (CMA minimizes, so the value
        # decreases), so the patience counter never reaches the threshold and
        # the full budget is used.
        cfg.cma_early_stop_patience = 2
        cfg.cma_early_stop_min_delta = 1e-2

        state = {"n": 0}

        def improving(x):
            state["n"] += 1
            return -float(state["n"])

        optimizer = make_optimizer(improving)
        strategy = make_strategy()
        optimizer._run_generations(strategy, 8)

        self.assertEqual(strategy.countiter, 8)

    def test_patience_never_exceeds_budget(self):
        cfg.cma_early_stop_patience = 50
        cfg.cma_early_stop_min_delta = 1.0

        optimizer = make_optimizer(quadratic)
        strategy = make_strategy()
        optimizer._run_generations(strategy, 5)

        self.assertEqual(strategy.countiter, 5)


if __name__ == "__main__":
    unittest.main()