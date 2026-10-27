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
            self.assertTrue(np.all(upper[3:] > lower[3:]))
        lower, upper = spline.bounds(3, 0.05)
        np.testing.assert_allclose(lower[:3], 0.0)
        np.testing.assert_allclose(upper[:3], 0.0)
        np.testing.assert_allclose(lower[3:9], -0.05)
        np.testing.assert_allclose(upper[3:9], 0.05)

    def test_mirror_convention(self) -> None:
        positions = np.array([[0.01, 0.0, 0.02]])
        yaw, pitch = np.array([0.3]), np.array([0.2])
        mirrored, yaw_mirrored, pitch_mirrored = spline.mirror(positions, yaw, pitch)
        np.testing.assert_allclose(mirrored, [[0.01, 0.0, -0.02]])
        np.testing.assert_allclose(yaw_mirrored, yaw)
        np.testing.assert_allclose(pitch_mirrored, -pitch)


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
