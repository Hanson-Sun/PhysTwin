"""CMA-ES optimize(): the selected parameters must be persisted regardless of
what the trailing validation rollout does."""
import os
import pickle
import tempfile
import unittest

from qqtt.engine.cma_optimize_warp import OptimizerCMA
from qqtt.utils import cfg


def make_optimizer(objective, error_func):
    """OptimizerCMA without the dataset/CUDA setup in __init__."""
    optimizer = OptimizerCMA.__new__(OptimizerCMA)
    optimizer.objective = objective
    optimizer.error_func = error_func
    return optimizer


def quadratic(x, visualize=False, video_path=None):
    return float(sum((v - 0.3) ** 2 for v in x))


class CmaResultPersistenceTests(unittest.TestCase):
    def setUp(self):
        self._base_dir = getattr(cfg, "base_dir", None)
        self._patience = cfg.cma_early_stop_patience
        self._min_delta = cfg.cma_early_stop_min_delta
        self._collision_dist = cfg.collision_dist
        self._tmp = tempfile.mkdtemp()
        cfg.base_dir = self._tmp
        # The Config default (0.06) normalizes outside the [0, 1] CMA bounds
        # for [0.01, 0.05]; the shipped YAMLs use 0.02.
        cfg.collision_dist = 0.02

    def tearDown(self):
        if self._base_dir is None:
            del cfg.base_dir
        else:
            cfg.base_dir = self._base_dir
        cfg.cma_early_stop_patience = self._patience
        cfg.cma_early_stop_min_delta = self._min_delta
        cfg.collision_dist = self._collision_dist

    def load_params(self):
        with open(os.path.join(self._tmp, "optimal_params.pkl"), "rb") as f:
            return pickle.load(f)

    def test_saved_when_validation_rollout_raises(self):
        def exploding_error_func(parameters, visualize=False, video_path=None):
            if visualize:
                raise FloatingPointError("simulation state became non-finite")
            return quadratic(parameters)

        make_optimizer(quadratic, exploding_error_func).optimize(max_iter=1)

        params = self.load_params()
        self.assertIn("global_spring_Y", params)
        self.assertIn("global_mass", params)

    def test_early_stop_still_saves(self):
        cfg.cma_early_stop_patience = 2
        cfg.cma_early_stop_min_delta = 1.0

        make_optimizer(quadratic, quadratic).optimize(max_iter=20)

        self.assertIn("global_spring_Y", self.load_params())

    def test_validation_rollout_still_runs(self):
        calls = []

        def recording_error_func(parameters, visualize=False, video_path=None):
            calls.append((visualize, video_path))
            return quadratic(parameters)

        make_optimizer(quadratic, recording_error_func).optimize(max_iter=1)

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0])
        self.assertTrue(calls[0][1].endswith("optimal.mp4"))


if __name__ == "__main__":
    unittest.main()