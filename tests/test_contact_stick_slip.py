import unittest

import numpy as np
import warp as wp

from qqtt.model.diff_simulator import contact as contact_law


@wp.kernel
def evaluate_stick_slip(
    displacement: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    dt: float,
    stiffness: wp.array(dtype=wp.float32),
    friction_limit: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
    next_displacement: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    friction_force, updated_displacement = contact_law.elastic_stick_slip(
        displacement[0], velocity[0], dt, stiffness[0], friction_limit[0]
    )
    force[0] = friction_force
    next_displacement[0] = updated_displacement
    loss[0] = friction_force[0]


class ContactStickSlipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.load_module(contact_law, device=wp.get_device("cpu"), recursive=True)

    def scalar(self, value, requires_grad=False):
        return wp.array(
            np.asarray([value], dtype=np.float32),
            dtype=wp.float32,
            device="cpu",
            requires_grad=requires_grad,
        )

    def vec3(self, value, requires_grad=False):
        return wp.array(
            np.asarray([value], dtype=np.float32),
            dtype=wp.vec3,
            device="cpu",
            requires_grad=requires_grad,
        )

    def launch(self, displacement, velocity, stiffness, friction_limit, requires_grad=False):
        stiffness_array = self.scalar(stiffness, requires_grad)
        friction_array = self.scalar(friction_limit, requires_grad)
        force = wp.empty(1, dtype=wp.vec3, device="cpu", requires_grad=requires_grad)
        next_displacement = wp.empty(
            1, dtype=wp.vec3, device="cpu", requires_grad=requires_grad
        )
        loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=requires_grad)
        wp.launch(
            evaluate_stick_slip,
            dim=1,
            inputs=[
                self.vec3(displacement, requires_grad),
                self.vec3(velocity, requires_grad),
                0.001,
                stiffness_array,
                friction_array,
            ],
            outputs=[force, next_displacement, loss],
            device="cpu",
        )
        return force, next_displacement, stiffness_array, friction_array, loss

    def test_static_force_exists_with_zero_relative_velocity(self):
        force, displacement, *_ = self.launch(
            (0.02, 0.0, 0.0), (0.0, 0.0, 0.0), 100.0, 2.5
        )
        np.testing.assert_allclose(force.numpy()[0], (-2.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(displacement.numpy()[0], (0.02, 0.0, 0.0), atol=1e-6)

    def test_force_builds_with_small_relative_motion(self):
        force, displacement, *_ = self.launch(
            (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), 100.0, 2.5
        )
        np.testing.assert_allclose(force.numpy()[0], (-0.01, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(displacement.numpy()[0], (0.0001, 0.0, 0.0), atol=1e-7)

    def test_sliding_force_saturates_and_return_maps_displacement(self):
        force, displacement, *_ = self.launch(
            (0.1, 0.0, 0.0), (0.0, 0.0, 0.0), 100.0, 2.5
        )
        self.assertAlmostEqual(float(np.linalg.norm(force.numpy()[0])), 2.5, places=5)
        self.assertAlmostEqual(float(displacement.numpy()[0, 0]), 0.025, places=5)

    def test_zero_friction_releases_stored_tangential_displacement(self):
        force, displacement, *_ = self.launch(
            (0.02, 0.0, 0.0), (0.0, 0.0, 0.0), 100.0, 0.0
        )
        np.testing.assert_allclose(force.numpy()[0], (0.0, 0.0, 0.0), atol=1e-7)
        np.testing.assert_allclose(displacement.numpy()[0], (0.0, 0.0, 0.0), atol=1e-7)

    def test_sliding_force_has_gradient_with_respect_to_coefficient(self):
        force, _, _, friction, loss = self.launch(
            (0.03, 0.0, 0.0), (0.0, 0.0, 0.0), 100.0, 2.0,
            requires_grad=True,
        )
        tape = wp.Tape()
        with tape:
            wp.launch(
                evaluate_stick_slip,
                dim=1,
                inputs=[
                    self.vec3((0.03, 0.0, 0.0), True),
                    self.vec3((0.0, 0.0, 0.0), True),
                    0.001,
                    self.scalar(100.0, True),
                    friction,
                ],
                outputs=[
                    force,
                    wp.empty(1, dtype=wp.vec3, device="cpu", requires_grad=True),
                    loss,
                ],
                device="cpu",
            )
        tape.backward(loss)
        self.assertTrue(np.isfinite(friction.grad.numpy()).all())
        self.assertGreater(abs(float(friction.grad.numpy()[0])), 1e-4)


if __name__ == "__main__":
    unittest.main()
