"""Rollout harness: stable carry-over, parameter writes, action swap."""

import unittest
from pathlib import Path

import numpy as np
import torch
import warp as wp

from asid.fim.case import load_case
from asid.fim.simulator import Rollout, SimConfig

CASE_DIR = Path("data/different_types/sim_soft_sloth_push")


@unittest.skipUnless((CASE_DIR / "final_data.pkl").is_file(), "case data not available")
class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_case(CASE_DIR, num_nodes=200, max_frames=12)

    def config(self, frames: int = 8) -> SimConfig:
        return SimConfig(frames=frames, num_substeps=100)

    def test_initial_state_and_motion(self):
        trajectory = Rollout(self.case, self.config()).run()
        self.assertEqual(tuple(trajectory.shape), (8, 200, 3))
        self.assertTrue(torch.isfinite(trajectory).all())
        np.testing.assert_allclose(
            trajectory[0].numpy(), self.case.nodes, atol=1e-6
        )
        self.assertGreater(float((trajectory[-1] - trajectory[0]).norm()), 1e-4)

    def test_repeatable_after_reset(self):
        rollout = Rollout(self.case, self.config())
        first = rollout.run()
        second = rollout.run()
        self.assertLess(float((first - second).abs().max()), 1e-5)

    def test_parameter_writes_change_trajectory(self):
        rollout = Rollout(self.case, self.config())
        nominal = rollout.run()
        rollout.set_spring_Y(torch.tensor(3e3))
        soft = rollout.run()
        self.assertGreater(float((nominal - soft).abs().max()), 1e-5)

    def test_soft_ground_force_changes_the_trajectory(self):
        base = Rollout(self.case, self.config()).run()
        no_spring = Rollout(
            self.case,
            SimConfig(frames=8, num_substeps=100, ground_stiffness=0.0),
        ).run()
        self.assertGreater(float((base - no_spring).abs().max()), 1e-6)

    def test_hard_ground_branch_still_runs(self):
        config = SimConfig(frames=4, num_substeps=100, soft_ground=False)
        trajectory = Rollout(self.case, config).run()
        self.assertEqual(tuple(trajectory.shape), (4, 200, 3))
        self.assertTrue(torch.isfinite(trajectory).all())

    def test_action_swap_matches_case_action(self):
        rollout = Rollout(self.case, self.config())
        nominal = rollout.run()
        rollout.set_action(self.case.controller.astype(np.float64), self.case.contact.astype(np.float64))
        swapped = rollout.run()
        self.assertLess(float((nominal - swapped).abs().max()), 1e-5)

    def test_carried_start_reproduces_a_fresh_run(self):
        rollout = Rollout(self.case, self.config())
        nominal = rollout.run()
        restart = rollout.run(initial=nominal[0])
        self.assertLess(float((nominal - restart).abs().max()), 1e-5)

    def test_carried_start_begins_at_the_given_state(self):
        rollout = Rollout(self.case, self.config(frames=12))
        nominal = rollout.run()
        start = nominal[6]
        carried = rollout.run(initial=start)
        self.assertEqual(tuple(carried.shape), (12, 200, 3))
        np.testing.assert_allclose(carried[0].numpy(), start.numpy(), atol=1e-5)
        self.assertTrue(torch.isfinite(carried).all())
        self.assertGreater(float((carried[-1] - carried[0]).norm()), 1e-4)

    def test_carried_start_is_repeatable(self):
        rollout = Rollout(self.case, self.config())
        start = rollout.run()[4]
        first = rollout.run(initial=start)
        second = Rollout(self.case, self.config()).run(initial=start)
        self.assertLess(float((first - second).abs().max()), 1e-5)

    def test_carried_start_shape_is_checked(self):
        rollout = Rollout(self.case, self.config())
        with self.assertRaises(ValueError):
            rollout.run(initial=np.zeros((3, 3), dtype=np.float32))

    def test_setters_write_the_simulator_arrays(self):
        rollout = Rollout(self.case, self.config())
        scale = rollout.sim.mass_normalization

        rollout.set_spring_Y(torch.tensor(2e4))
        stored = torch.exp(wp.to_torch(rollout.sim.wp_spring_Y, requires_grad=False))
        np.testing.assert_allclose(stored.cpu(), 2e4 * scale, rtol=1e-5)

        masses = torch.full((200,), 0.002)
        rollout.set_masses(masses)
        stored = wp.to_torch(rollout.sim.wp_masses, requires_grad=False)
        np.testing.assert_allclose(stored.cpu(), masses, rtol=1e-6)

        rollout.set_dashpot(50.0)
        stored = wp.to_torch(rollout.sim.wp_dashpot_damping, requires_grad=False)
        np.testing.assert_allclose(stored.cpu(), [50.0 * scale], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
