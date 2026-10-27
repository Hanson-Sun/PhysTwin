"""Spline parameterization, the envelope that bounds it, and the search inside."""

import dataclasses
import unittest

import numpy as np

from asid.planning import spline
from asid.planning.envelope import Envelope, within_workspace
from asid.planning.search import plan_spline


def _template(frames: int = 5, points: int = 4):
    rng = np.random.default_rng(0)
    controller = rng.normal(size=(frames, points, 3)) * 0.01
    contact = rng.normal(size=(frames, 2 * points, 3)) * 0.01
    return controller, contact


def _envelope(template, frames: int, offset_limit: float = 0.05) -> Envelope:
    """A permissive envelope: wide box, loose speed, always in contact."""
    nodes = np.asarray(template[1], dtype=float).reshape(-1, 3)
    return Envelope(
        lower=np.full(3, -1.0),
        upper=np.full(3, 1.0),
        max_speed=10.0,
        offset_limit=offset_limit,
        orientation_limit=spline.ORIENTATION_LIMIT,
        max_rate=spline.MAX_ANGULAR_RATE,
        dt=1.0 / 30.0,
        contact_radius=10.0,
        nodes=nodes,
        frames=frames,
    )


class SpecTest(unittest.TestCase):
    def test_start_reproduces_template(self) -> None:
        controller, contact = _template()
        motion = spline.motion(spline.Spec.start(3), controller, contact, 5)
        np.testing.assert_allclose(motion.controller, controller, atol=1e-12)
        np.testing.assert_allclose(motion.contact, contact, atol=1e-12)

    def test_offsets_at_knots(self) -> None:
        controller, contact = _template()
        spec = spline.Spec(
            np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]]),
            np.array([0.0, 0.5, 1.0]),
            np.zeros(3),
            np.zeros(3),
        )
        motion = spline.motion(spec, controller, contact, 5)
        np.testing.assert_allclose(
            motion.controller.mean(axis=1)[::2] - controller.mean(axis=1)[::2],
            spec.positions,
            atol=1e-12,
        )

    def test_rotation_is_rigid(self) -> None:
        controller, _ = _template()
        spec = spline.Spec(
            np.zeros((3, 3)),
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.3, -0.2]),
            np.array([0.0, -0.1, 0.15]),
        )
        motion = spline.motion(spec, controller, controller, 5)
        for frame in range(5):
            before = np.linalg.norm(
                controller[frame][:, None] - controller[frame][None], axis=-1
            )
            after = np.linalg.norm(
                motion.controller[frame][:, None] - motion.controller[frame][None], axis=-1
            )
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_round_trip_and_order(self) -> None:
        vector = np.arange(6 * 4 - 2, dtype=float) * 0.001
        spec = spline.unflatten(vector, 4)
        self.assertTrue(np.all(np.diff(spec.times) > 0))
        self.assertEqual(spec.times[0], 0.0)
        self.assertEqual(spec.times[-1], 1.0)
        np.testing.assert_allclose(spline.flatten(spec), vector)

    def test_unflatten_rejects_wrong_size(self) -> None:
        with self.assertRaises(ValueError):
            spline.unflatten(np.zeros(3), 3)

    def test_mirror_convention(self) -> None:
        positions = np.array([[0.01, 0.0, 0.02]])
        yaw, pitch = np.array([0.3]), np.array([0.2])
        mirrored, yaw_mirrored, pitch_mirrored = spline.mirror(positions, yaw, pitch)
        np.testing.assert_allclose(mirrored, [[0.01, 0.0, -0.02]])
        np.testing.assert_allclose(yaw_mirrored, yaw)
        np.testing.assert_allclose(pitch_mirrored, -pitch)


class EnvelopeTest(unittest.TestCase):
    def test_bounds_pin_the_start_pose(self) -> None:
        envelope = _envelope(_template(), frames=5)
        for knots in (2, 3, 5):
            lower, upper = envelope.bounds(knots)
            self.assertEqual(lower.shape, (6 * knots - 2,))
            pinned = np.zeros(lower.size, dtype=bool)
            pinned[[0, 1, 2, 4 * knots - 2, 5 * knots - 2]] = True
            self.assertTrue(np.all(upper[pinned] == lower[pinned]))
            self.assertTrue(np.all(upper[~pinned] > lower[~pinned]))
        lower, upper = envelope.bounds(3)
        np.testing.assert_allclose(lower[:3], 0.0)
        np.testing.assert_allclose(upper[:3], 0.0)
        np.testing.assert_allclose(lower[3:9], -0.05)
        np.testing.assert_allclose(upper[3:9], 0.05)
        # The wrist also starts where the tool is: first yaw and pitch pinned,
        # later waypoints free to rotate.
        np.testing.assert_allclose(lower[10:13], [0.0, -np.deg2rad(30.0), -np.deg2rad(30.0)])
        np.testing.assert_allclose(upper[10:13], [0.0, np.deg2rad(30.0), np.deg2rad(30.0)])

    def test_from_demo_pads_the_recorded_motion(self) -> None:
        template = _template(frames=5)
        nodes = np.asarray(template[1], dtype=float).reshape(-1, 3)
        envelope = Envelope.from_demo(nodes, template, frames=5, dt=1.0 / 30.0, contact_radius=0.02)
        occupied = np.concatenate([nodes, template[0].reshape(-1, 3), template[1].reshape(-1, 3)])
        np.testing.assert_allclose(envelope.lower, occupied.min(axis=0) - 0.15)
        np.testing.assert_allclose(envelope.upper, occupied.max(axis=0) + 0.15)
        peak = max(
            float(np.linalg.norm(np.diff(cloud, axis=0), axis=-1).max())
            for cloud in template
        )
        self.assertAlmostEqual(envelope.max_speed, 1.5 * peak / (1.0 / 30.0))

    def test_contact_guard_rejects_a_miss(self) -> None:
        """The score is flat before contact, so a miss must be infeasible."""
        template = _template(frames=5)
        far = dataclasses.replace(
            _envelope(template, frames=5),
            contact_radius=1e-6,
            nodes=np.full((2, 3), 9.0),
        )
        self.assertIsNone(far.admit(spline.Spec.start(3), template))
        self.assertIsNotNone(_envelope(template, frames=5).admit(spline.Spec.start(3), template))

    def test_workspace_check_uses_bounds_and_speed(self) -> None:
        points = np.zeros((3, 2, 3))
        self.assertTrue(within_workspace(points, points, np.full(3, -1.0), np.full(3, 1.0), 1.0, 0.1))
        outside = points.copy()
        outside[1, 0, 0] = 2.0
        self.assertFalse(within_workspace(outside, points, np.full(3, -1.0), np.full(3, 1.0), 1.0, 0.1))
        with self.assertRaises(ValueError):
            within_workspace(points, points, np.ones(3), np.zeros(3), 1.0, 0.1)


class PlanTest(unittest.TestCase):
    """The searched plan must obey the envelope *as sampled*, not only at knots."""

    def _plan(self, offset_limit: float = 0.05):
        template = _template(frames=8, points=4)
        envelope = _envelope(template, frames=8, offset_limit=offset_limit)
        scored = []

        def score(candidate, contact_points):
            scored.append(1.0)
            return 1.0

        plan = plan_spline(
            score, template, envelope, knots=3, samples=6, iterations=2, seed=0
        )
        return plan, scored

    def test_plan_respects_the_sampled_envelope(self) -> None:
        plan, _ = self._plan()
        positions, yaw, pitch = plan.motion.positions, plan.motion.yaw, plan.motion.pitch
        self.assertLessEqual(float(np.abs(positions).max()), 0.05 + 1e-12)
        self.assertLessEqual(
            float(max(np.abs(yaw).max(), np.abs(pitch).max())), np.deg2rad(30.0) + 1e-12
        )
        self.assertLessEqual(
            float(max(np.abs(np.diff(yaw)).max(), np.abs(np.diff(pitch)).max())),
            spline.MAX_ANGULAR_RATE / 30.0 + 1e-12,
        )
        self.assertEqual(yaw[0], 0.0)
        self.assertEqual(pitch[0], 0.0)
        self.assertEqual(float(np.abs(positions[0]).max()), 0.0)
        self.assertEqual(plan.diagnostics.max_offset, float(np.abs(positions).max()))

    def test_search_improves_while_staying_in_the_envelope(self) -> None:
        """Guards the regression where every sample was rejected and CEM stalled.

        Measured on this objective: with a natural cubic interpolant the search
        returns the initializer unchanged (-0.000845), because the overshoot puts
        every candidate outside the envelope; with the shape-preserving spline it
        reaches 0.0352 while the offsets stay inside the box.
        """
        template = _template(frames=60, points=4)
        envelope = _envelope(template, frames=60)

        def score(candidate, contact_points):
            return float(candidate.mean(axis=1)[:, 0].mean())

        start = score(*template)
        plan = plan_spline(
            score, template, envelope, knots=3, samples=6, iterations=4, seed=0
        )
        self.assertGreater(plan.result.score, start + 1e-3)
        self.assertLessEqual(float(np.abs(plan.motion.positions).max()), 0.05 + 1e-12)
        self.assertLessEqual(
            float(max(np.abs(plan.motion.yaw).max(), np.abs(plan.motion.pitch).max())),
            np.deg2rad(30.0) + 1e-12,
        )

    def test_out_of_envelope_samples_are_rejected(self) -> None:
        plan, scored = self._plan(offset_limit=0.005)
        self.assertLess(len(scored), plan.result.evaluations)
        self.assertLessEqual(float(np.abs(plan.motion.positions).max()), 0.005 + 1e-12)


class ReplayTest(unittest.TestCase):
    """The MuJoCo replay must apply the same mirror as :func:`spline.mirror`."""

    def test_yaw_keeps_sign_and_pitch_flips(self) -> None:
        from mujoco_sim.interactor import apply_tool_motion

        trajectory = [((0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0))] * 4
        yawed = apply_tool_motion(
            trajectory, np.zeros((4, 3)), yaw=np.full(4, 0.4), pitch=np.zeros(4)
        )
        w, x, y, z = yawed[0][1]
        self.assertAlmostEqual(z, np.sin(0.2), places=9)
        self.assertAlmostEqual(w, np.cos(0.2), places=9)

        pitched = apply_tool_motion(
            trajectory, np.zeros((4, 3)), yaw=np.zeros(4), pitch=np.full(4, 0.4)
        )
        w, x, y, z = pitched[0][1]
        self.assertAlmostEqual(y, -np.sin(0.2), places=9)
        self.assertAlmostEqual(w, np.cos(0.2), places=9)

    def test_position_reflects_z(self) -> None:
        from mujoco_sim.interactor import apply_tool_motion

        trajectory = [((0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0))] * 3
        offsets = np.tile([0.0, 0.0, 0.03], (3, 1))
        out = apply_tool_motion(trajectory, offsets)
        self.assertAlmostEqual(out[0][0][2], 0.47)
        self.assertAlmostEqual(out[0][0][0], 0.0)


if __name__ == "__main__":
    unittest.main()
