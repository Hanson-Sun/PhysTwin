"""Focused tests for bounded trajectory selection and FIM scoring."""

import unittest

import numpy as np

from asid.fim.operators import Observation
from asid.planning.trajectory import fisher_score, make_trajectory, plan_trajectory


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.controller = np.zeros((4, 1, 3))
        self.contact = np.zeros((4, 1, 3))
        self.options = dict(
            lower=np.full(3, -1.0), upper=np.ones(3), max_speed=100.0,
            dt=0.1, offset_limit=0.2, knots=1,
        )

    def test_sampled_selection_returns_best_feasible(self):
        score = lambda t: float(t.offsets[-1, 0])
        plan = plan_trajectory(self.controller, self.contact, score, candidates=8, seed=2, **self.options)
        self.assertAlmostEqual(plan.score, plan.sampled_scores.max())
        self.assertAlmostEqual(plan.score, score(plan.trajectory))
        np.testing.assert_array_equal(plan.trajectory.offsets[0], np.zeros(3))
        self.assertTrue(np.all(np.abs(plan.controls) <= 0.2))

    def test_gradient_refinement_improves_smooth_objective(self):
        score = lambda t: float(t.offsets[-1, 0])
        plan = plan_trajectory(
            self.controller, self.contact, score, candidates=1,
            gradient_steps=2, action_step=0.01, **self.options,
        )
        self.assertGreater(plan.score, plan.sampled_scores.max())
        self.assertEqual(plan.gradient_status, "improved with checked finite-difference gradient")

    def test_unstable_gradient_preserves_sampled_result(self):
        def score(t):
            x = t.offsets[-1, 0]
            return float(x * x * x)
        plan = plan_trajectory(
            self.controller, self.contact, score, candidates=1,
            gradient_steps=1, action_step=0.01, gradient_tolerance=0.2, **self.options,
        )
        self.assertEqual(plan.gradient_status, "unstable finite-difference gradient")
        self.assertEqual(plan.score, plan.sampled_scores.max())

    def test_fim_score_conditions_on_existing_information(self):
        class Rollout:
            def set_action(self, controller, contact):
                self.scale = controller[-1, 0, 0]
            def run(self, initial=None):
                import torch
                return torch.tensor([[[self.scale * self.phi[0], self.phi[1], 0.0]]])
        class Partition:
            def apply(self, rollout, phi):
                rollout.phi = phi
        trajectory = make_trajectory(self.controller, self.contact, np.array([[0.1, 0, 0]]))
        args = dict(
            rollout=Rollout(), partition=Partition(), observation=Observation(np.array([0]), 1.0),
            trajectory=trajectory, parameter_samples=[np.zeros(2)], weights=np.ones(1),
            prior_std=np.ones(2), rank=2, parameter_step=1e-3, seed=1,
        )
        initial = fisher_score(current_factors=None, **args)
        conditioned = fisher_score(current_factors=2 * np.eye(2), **args)
        self.assertGreater(initial, conditioned)
        self.assertGreater(conditioned, 0)


if __name__ == "__main__":
    unittest.main()
