"""Spline parameterization: sampling, ordering, bounds and the MuJoCo mirror."""

import unittest

import numpy as np

from asid.planning import spline


def _template(frames: int = 5, points: int = 4):
    rng = np.random.default_rng(0)
    controller = rng.normal(size=(frames, points, 3)) * 0.01
    contact = rng.normal(size=(frames, 2 * points, 3)) * 0.01
    return controller, contact


class SpecTest(unittest.TestCase):
    def test_start_reproduces_template(self) -> None:
        controller, contact = _template()
        out_controller, out_contact = spline.sample(
            spline.Spec.start(3), controller, contact, 5
        )
        np.testing.assert_allclose(out_controller, controller, atol=1e-12)
        np.testing.assert_allclose(out_contact, contact, atol=1e-12)

    def test_offsets_at_knots(self) -> None:
        controller, contact = _template()
        spec = spline.Spec(
            np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]]),
            np.array([0.0, 0.5, 1.0]),
            np.zeros(3),
            np.zeros(3),
        )
        out, _ = spline.sample(spec, controller, contact, 5)
        np.testing.assert_allclose(
            out.mean(axis=1)[::2] - controller.mean(axis=1)[::2],
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
        out, _ = spline.sample(spec, controller, controller, 5)
        for frame in range(5):
            before = np.linalg.norm(
                controller[frame][:, None] - controller[frame][None], axis=-1
            )
            after = np.linalg.norm(out[frame][:, None] - out[frame][None], axis=-1)
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

    def test_bounds_shape(self) -> None:
        for knots in (2, 3, 5):
            lower, upper = spline.bounds(knots, 0.05)
            self.assertEqual(lower.shape, (6 * knots - 2,))
            pinned = np.zeros(lower.size, dtype=bool)
            pinned[[0, 1, 2, 4 * knots - 2, 5 * knots - 2]] = True
            self.assertTrue(np.all(upper[pinned] == lower[pinned]))
            self.assertTrue(np.all(upper[~pinned] > lower[~pinned]))
        lower, upper = spline.bounds(3, 0.05)
        np.testing.assert_allclose(lower[:3], 0.0)
        np.testing.assert_allclose(upper[:3], 0.0)
        np.testing.assert_allclose(lower[3:9], -0.05)
        np.testing.assert_allclose(upper[3:9], 0.05)
        # The wrist also starts where the tool is: first yaw and pitch pinned,
        # later waypoints free to rotate.
        np.testing.assert_allclose(lower[10:13], [0.0, -np.deg2rad(30.0), -np.deg2rad(30.0)])
        np.testing.assert_allclose(upper[10:13], [0.0, np.deg2rad(30.0), np.deg2rad(30.0)])

    def test_mirror_convention(self) -> None:
        positions = np.array([[0.01, 0.0, 0.02]])
        yaw, pitch = np.array([0.3]), np.array([0.2])
        mirrored, yaw_mirrored, pitch_mirrored = spline.mirror(positions, yaw, pitch)
        np.testing.assert_allclose(mirrored, [[0.01, 0.0, -0.02]])
        np.testing.assert_allclose(yaw_mirrored, yaw)
        np.testing.assert_allclose(pitch_mirrored, -pitch)


class PlanTest(unittest.TestCase):
    """The searched plan must obey the envelope *as sampled*, not only at knots."""

    def _plan(self, offset_limit: float = 0.05):
        controller, contact = _template(frames=8, points=4)
        nodes = np.asarray(contact, dtype=float).reshape(-1, 3)
        scored = []

        def score(candidate, contact_points):
            scored.append(1.0)
            return 1.0

        plan = spline.plan_spline(
            score,
            (controller, contact),
            nodes,
            frames=8,
            lower=np.full(3, -1.0),
            upper=np.full(3, 1.0),
            max_speed=10.0,
            dt=1.0 / 30.0,
            contact_radius=10.0,
            knots=3,
            offset_limit=offset_limit,
            samples=6,
            iterations=2,
            elites=2,
            seed=0,
        )
        return plan, scored

    def test_plan_respects_the_sampled_envelope(self) -> None:
        plan, _ = self._plan()
        positions, yaw, pitch = spline.controls(plan.spec, 8)
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

    def test_search_improves_while_staying_in_the_envelope(self) -> None:
        """Guards the regression where every sample was rejected and CEM stalled.

        Measured on this objective: with a natural cubic interpolant the search
        returns the initializer unchanged (-0.000845), because the overshoot puts
        every candidate outside the envelope; with the shape-preserving spline it
        reaches 0.0352 while the offsets stay inside the box.
        """
        frames = 60
        controller, contact = _template(frames=frames, points=4)
        nodes = np.asarray(contact, dtype=float).reshape(-1, 3)

        def score(candidate, contact_points):
            return float(candidate.mean(axis=1)[:, 0].mean())

        start = score(controller, contact)
        plan = spline.plan_spline(
            score,
            (controller, contact),
            nodes,
            frames=frames,
            lower=np.full(3, -1.0),
            upper=np.full(3, 1.0),
            max_speed=10.0,
            dt=1.0 / 30.0,
            contact_radius=10.0,
            knots=3,
            offset_limit=0.05,
            samples=6,
            iterations=4,
            elites=2,
            seed=0,
        )
        self.assertGreater(plan.result.score, start + 1e-3)
        positions, yaw, pitch = spline.controls(plan.spec, frames)
        self.assertLessEqual(float(np.abs(positions).max()), 0.05 + 1e-12)
        self.assertLessEqual(
            float(max(np.abs(yaw).max(), np.abs(pitch).max())), np.deg2rad(30.0) + 1e-12
        )

    def test_out_of_envelope_samples_are_rejected(self) -> None:
        plan, scored = self._plan(offset_limit=0.005)
        self.assertLess(len(scored), plan.result.evaluations)
        positions, _, _ = spline.controls(plan.spec, 8)
        self.assertLessEqual(float(np.abs(positions).max()), 0.005 + 1e-12)


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
