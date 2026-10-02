"""Velocity-level Coulomb friction: the properties the law must hold.

These are the checks the reverted bristle law failed (413 mm of drift in a
single frame, 299% off the analytic deceleration in sliding, and never locking
on to a moving surface), so they are the regression guard for that revert.
"""

import unittest

import numpy as np
import warp as wp

from qqtt.model.diff_simulator import contact as contact_law
from qqtt.model.diff_simulator import spring_mass_warp as smw

# The simulator's own operating point.
DT = 5e-5
STEPS = 667  # one frame at 30 FPS
MASS = 0.01
N_LOAD = 30.0
MU = 0.3
# Importing spring_mass_warp sets the default device, so the contact-kernel test
# must run there too or the HashGrid and the arrays land on different devices.
SIM_DEVICE = "cuda:0"


def _upstream_module():
    from qqtt.model.diff_simulator import spring_mass_warp_upstream
    return spring_mass_warp_upstream


def _source_of(module):
    """Read a module's source from disk.

    ``inspect.getsource`` is unavailable here because the package is imported
    from an installed/compiled location in some environments.
    """
    import importlib.util
    import os
    spec = importlib.util.find_spec(module.__name__)
    path = spec.origin
    if not os.path.exists(path):
        raise unittest.SkipTest(f"source not available for {module.__name__}")
    with open(path) as fh:
        return fh.read()


def _slice_class(src, header, stop=None):
    """Slice from `header` up to `stop` (or the next top-level `class`)."""
    start = src.index(header)
    if stop is None:
        rest = src[start + len(header):]
        stop = "\nclass "
        end = rest.find(stop)
        return src[start:start + len(header) + (end if end != -1 else len(rest))]
    end = src.index(stop, start + len(header))
    return src[start:end]


def _slice_method(src, header, stop):
    """Slice from `header` up to `stop`, or to the end of `src` if absent."""
    start = src.index(header)
    end = src.find(stop, start + len(header))
    return src[start:] if end == -1 else src[start:end]


@wp.kernel
def smw_integrate_passthrough(
    v_in: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    v_out[0] = v_in[0]


@wp.kernel
def project_once(
    v_rel: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    load: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    mu: wp.array(dtype=wp.float32),
    dt: float,
    v_out: wp.array(dtype=wp.vec3),
):
    v_out[0] = contact_law.coulomb_project(
        v_rel[0], normal[0], load[0], mass[0], mu[0], dt
    )


@wp.kernel
def step_held(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    load: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    mu: wp.array(dtype=wp.float32),
    a_ext: float,
    dt: float,
    v_out: wp.array(dtype=wp.vec3),
    x_out: wp.array(dtype=wp.vec3),
):
    """One substep: accumulate an external tangential load, then apply friction."""
    v_free = v[0] + wp.vec3(a_ext * dt, 0.0, 0.0)
    v_new = contact_law.coulomb_project(v_free, normal[0], load[0], mass[0], mu[0], dt)
    v_out[0] = v_new
    x_out[0] = x[0] + v_new * dt


@wp.kernel
def step_surface(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    load: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    mu: wp.array(dtype=wp.float32),
    v_surface: float,
    dt: float,
    v_out: wp.array(dtype=wp.vec3),
    x_out: wp.array(dtype=wp.vec3),
):
    """One substep against a surface moving at constant ``v_surface``."""
    v_new = contact_law.coulomb_project(
        v[0] - wp.vec3(v_surface, 0.0, 0.0), normal[0], load[0], mass[0], mu[0], dt
    ) + wp.vec3(v_surface, 0.0, 0.0)
    v_out[0] = v_new
    x_out[0] = x[0] + v_new * dt


@wp.kernel
def make_loss(v: wp.array(dtype=wp.vec3), loss: wp.array(dtype=wp.float32)):
    loss[0] = v[0][0]


def scalar(value, device="cpu"):
    return wp.array(np.array([value], np.float32), dtype=wp.float32, device=device)


def vec3(value, device="cpu"):
    return wp.array(np.array([value], np.float32), dtype=wp.vec3, device=device)


def zeros(device="cpu"):
    return wp.zeros(1, dtype=wp.vec3, device=device)


class CoulombFrictionLawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.load_module(contact_law, device=wp.get_device("cpu"), recursive=True)

    def setUp(self):
        self.normal = vec3((0.0, 0.0, 1.0))
        self.load = scalar(N_LOAD)
        self.mass = scalar(MASS)
        self.mu = scalar(MU)
        self.v = zeros()
        self.x = zeros()

    def _project(self, v_rel, mu=None, load=None):
        out = zeros()
        wp.launch(
            project_once, dim=1,
            inputs=[vec3(v_rel), self.normal,
                    self.load if load is None else load,
                    self.mass, self.mu if mu is None else mu, DT],
            outputs=[out], device="cpu",
        )
        return out.numpy()[0]

    def _hold(self, steps, a_ext, dt=DT):
        for _ in range(steps):
            v_new, x_new = zeros(), zeros()
            wp.launch(
                step_held, dim=1,
                inputs=[self.x, self.v, self.normal, self.load, self.mass, self.mu,
                        a_ext, dt],
                outputs=[v_new, x_new], device="cpu",
            )
            self.v, self.x = v_new, x_new
        return self.v.numpy()[0].copy(), self.x.numpy()[0].copy()

    def _ride(self, steps, v_surface, dt=DT):
        for _ in range(steps):
            v_new, x_new = zeros(), zeros()
            wp.launch(
                step_surface, dim=1,
                inputs=[self.x, self.v, self.normal, self.load, self.mass, self.mu,
                        v_surface, dt],
                outputs=[v_new, x_new], device="cpu",
            )
            self.v, self.x = v_new, x_new
            yield self.v.numpy()[0][0] - v_surface

    def test_stick_below_the_cone_does_not_drift(self):
        # A sub-cone tangential load must be held exactly. Any law that
        # under-removes velocity settles at a nonzero creep fixed point.
        _, x = self._hold(STEPS, a_ext=0.9 * MU * N_LOAD / MASS)
        self.assertAlmostEqual(x[0], 0.0, places=9)

    def test_stick_is_timestep_independent(self):
        # Physics must not depend on dt; the reverted tanh law scaled with it.
        a_ext = 0.9 * MU * N_LOAD / MASS
        _, x_dt = self._hold(STEPS, a_ext, dt=DT)
        self.v, self.x = zeros(), zeros()
        _, x_half = self._hold(2 * STEPS, a_ext, dt=DT / 2)
        self.assertAlmostEqual(x_dt[0], 0.0, places=9)
        self.assertAlmostEqual(x_dt[0], x_half[0], places=9)

    def test_slip_decelerates_at_the_analytic_coulomb_rate(self):
        a_ext = 1.5 * MU * N_LOAD / MASS
        expected_accel = a_ext - MU * N_LOAD / MASS
        v, _ = self._hold(STEPS, a_ext=a_ext)
        # In sustained slip friction removes exactly mu*N/m, so the mean
        # acceleration over the run must be the analytic net force over mass.
        measured_accel = v[0] / (STEPS * DT)
        self.assertAlmostEqual(measured_accel / expected_accel, 1.0, places=3)

    def test_zero_friction_is_the_identity(self):
        v_rel = (0.3, -0.2, 0.0)
        np.testing.assert_allclose(self._project(v_rel, mu=scalar(0.0)), v_rel, atol=1e-7)

    def test_zero_normal_load_is_the_identity(self):
        v_rel = (0.3, -0.2, 0.0)
        np.testing.assert_allclose(self._project(v_rel, load=scalar(0.0)), v_rel, atol=1e-7)

    def test_normal_component_is_untouched(self):
        v_rel = (0.0, 0.0, 0.5)
        np.testing.assert_allclose(self._project(v_rel), v_rel, atol=1e-7)

    def test_stick_cancels_all_tangential_velocity(self):
        # below the cone the whole tangential component is removed
        limit = MU * N_LOAD * DT / MASS
        result = self._project((limit * 0.5, -limit * 0.25, 0.4))
        self.assertAlmostEqual(result[0], 0.0, places=7)
        self.assertAlmostEqual(result[1], 0.0, places=7)
        self.assertAlmostEqual(result[2], 0.4, places=7)

    def test_locks_onto_a_moving_surface_without_oscillating(self):
        v_surface = 0.1
        rels = list(self._ride(STEPS, v_surface))
        self.assertAlmostEqual(rels[-1], 0.0, places=6)
        # no ringing: once locked the relative velocity must never change sign
        locked = [r for r in rels if abs(r) < 1e-3 * v_surface]
        self.assertTrue(locked, "never locked on")
        signs = [np.sign(r) for r in locked if r != 0.0]
        self.assertLessEqual(len(set(signs)), 1)


class FrictionKernelTapeTests(unittest.TestCase):
    """The new kernels must be differentiable, including the chained launch.

    Friction is applied as a velocity constraint between two launches, so the
    adjoint has to flow through apply_controller_friction and then
    apply_ground_friction. That chain is the part most likely to silently
    produce wrong gradients.

    Note: ``controller_contact_friction`` is a plain float on this simulator,
    so there is no gradient with respect to mu here. Only the upstream
    simulator exposes a learnable friction array, and mu is searched with CMA
    (derivative-free) regardless.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.load_module(contact_law, device=wp.get_device(SIM_DEVICE), recursive=True)
        wp.load_module(smw, device=wp.get_device(SIM_DEVICE), recursive=True)

    def _chain_grad(self, v_rel):
        dev = SIM_DEVICE
        v_in = vec3(v_rel, dev)
        v_in.requires_grad = True
        mid = wp.zeros(1, dtype=wp.vec3, device=dev, requires_grad=True)
        out = wp.zeros(1, dtype=wp.vec3, device=dev, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                smw.apply_controller_friction, dim=1,
                inputs=[v_in, scalar(MASS, dev), 1, vec3((0.0, 0.0, 1.0), dev),
                        scalar(N_LOAD, dev), vec3((0.0, 0.0, 0.0), dev), MU, DT],
                outputs=[mid], device=dev,
            )
            wp.launch(
                smw.apply_ground_friction, dim=1,
                inputs=[mid, vec3((0.0, 0.0, 0.0), dev), scalar(MASS, dev), 1,
                        scalar(0.5, dev), DT, -1.0, 2e-5],
                outputs=[out], device=dev,
            )
            wp.launch(make_loss, dim=1, inputs=[out], outputs=[loss], device=dev)
        tape.backward(loss)
        return v_in.grad.numpy()[0].copy()

    def test_chained_backward_produces_finite_gradients(self):
        grad = self._chain_grad((0.3, -0.2, 0.0))
        self.assertTrue(np.isfinite(grad).all(), f"non-finite velocity grad {grad}")
        self.assertGreater(np.abs(grad).max(), 0.0)

    def test_backward_is_deterministic(self):
        # a second identical pass must reproduce the adjoint exactly
        np.testing.assert_allclose(self._chain_grad((0.3, -0.2, 0.0)),
                                   self._chain_grad((0.3, -0.2, 0.0)), rtol=1e-6)


class PerSubstepBufferTests(unittest.TestCase):
    """Friction intermediates must be per-substep, not shared across substeps.

    Regression guard for a real NaN-gradient failure. The first implementation
    allocated ``wp_v_after_controller_friction`` once on the simulator and
    wrote it from every substep of ``step()``. All 667 writes land on one tape,
    so the adjoint of the shared array accumulates ``2**num_substeps`` and
    overflows float32 to ``inf`` partway through the backward pass -- the
    forward simulation stays perfectly finite while ``spring_Y.grad`` comes
    back all-NaN. These tests pin the invariant at the level where it broke.
    """

    NUM_SUBSTEPS = 667

    @staticmethod
    def _chain(nsub, shared_output):
        """Run `nsub` identity launches, optionally writing one shared array."""
        dev = SIM_DEVICE
        v_in = wp.array(np.array([[0.3, -0.2, 0.1]], np.float32),
                        dtype=wp.vec3, device=dev)
        v_in.requires_grad = True
        n = 1
        if shared_output:
            buffers = [wp.zeros(n, dtype=wp.vec3, device=dev, requires_grad=True)] * nsub
        else:
            buffers = [wp.zeros(n, dtype=wp.vec3, device=dev, requires_grad=True)
                       for _ in range(nsub)]
        loss = wp.zeros(1, dtype=wp.float32, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            cur = v_in
            for s in range(nsub):
                wp.launch(smw_integrate_passthrough, dim=n,
                          inputs=[cur], outputs=[buffers[s]], device=dev)
                cur = buffers[s]
            wp.launch(make_loss, dim=n, inputs=[cur], outputs=[loss], device=dev)
        tape.backward(loss)
        return v_in.grad.numpy()[0].copy()

    def test_shared_output_buffer_overflows_the_adjoint(self):
        """Documents the failure mode: the shared layout really does blow up."""
        shared = self._chain(self.NUM_SUBSTEPS, shared_output=True)
        self.assertFalse(np.isfinite(shared).all(),
                         "expected the shared-buffer layout to overflow; if this "
                         "now passes, the test no longer reproduces the bug")

    def test_per_substep_buffers_keep_the_adjoint_finite(self):
        per_step = self._chain(self.NUM_SUBSTEPS, shared_output=False)
        self.assertTrue(np.isfinite(per_step).all(),
                        f"non-finite gradient from per-substep buffers: {per_step}")

    def test_simulator_allocates_friction_buffers_per_substep(self):
        """Both simulators must keep the friction stages inside State."""
        for mod in (smw, _upstream_module()):
            src = _source_of(mod)
            state_src = src[src.index("class State"):src.index("class SpringMassSystemWarp")]
            self.assertIn("wp_v_after_controller_friction", state_src)
            self.assertIn("wp_v_after_ground_friction", state_src)
            sim_src = src[src.index("class SpringMassSystemWarp"):]
            init_src = _slice_method(sim_src, "    def __init__(", stop="\n    def ")
            # NOT as simulator-level attributes shared by every substep
            self.assertNotIn(
                "self.wp_v_after_controller_friction = wp.zeros", init_src,
                "controller friction buffer must not be shared across substeps")
            self.assertNotIn(
                "self.wp_v_after_ground_friction = wp.zeros", init_src,
                "ground friction buffer must not be shared across substeps")

    def test_step_writes_friction_buffers_per_substep(self):
        """step() must index the buffers by substep, not use the bare attribute."""
        for mod in (smw, _upstream_module()):
            src = _source_of(mod)
            step_src = _slice_method(
                src[src.index("class SpringMassSystemWarp"):],
                "    def step(self)", stop="\n    def ")
            self.assertIn("self.wp_states[i].wp_v_after_controller_friction", step_src)
            self.assertIn("self.wp_states[i].wp_v_after_ground_friction", step_src)


class DegenerateContactNormalTests(unittest.TestCase):
    """A symmetric pinch on one particle must not produce friction at all."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.load_module(contact_law, device=wp.get_device(SIM_DEVICE), recursive=True)
        wp.load_module(smw, device=wp.get_device(SIM_DEVICE), recursive=True)

    def test_opposed_equal_contacts_give_zero_friction(self):
        radius = 0.014
        dev = SIM_DEVICE
        # one object particle with two controller points pressing from both
        # sides at equal distance, so the load-weighted normal cancels exactly
        x = vec3((0.0, 0.0, 0.0), dev)
        ctrl_x = wp.array(
            np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]], np.float32),
            dtype=wp.vec3, device=dev,
        )
        ctrl_v = wp.zeros(2, dtype=wp.vec3, device=dev)
        active = wp.zeros(1, dtype=wp.int32, device=dev)
        count = wp.zeros(1, dtype=wp.int32, device=dev)
        total = wp.zeros(1, dtype=wp.int32, device=dev)
        forces = zeros(dev)
        normal_out = zeros(dev)
        load_out = wp.zeros(1, dtype=wp.float32, device=dev)
        surf_out = zeros(dev)

        grid = wp.HashGrid(16, 16, 16, device=dev)
        grid.build(ctrl_x, radius)

        wp.launch(
            smw.batched_controller_contact_force, dim=1,
            inputs=[x, zeros(dev), ctrl_x, ctrl_v, grid.id, 1, 2, radius, radius,
                    radius, 3e4, active, count, total],
            outputs=[forces, normal_out, load_out, surf_out], device=dev,
        )
        # both contacts really are active, so the normal load is not trivially 0
        self.assertGreater(int(count.numpy()[0]), 0)
        # ...but the published friction frame is zeroed, so coulomb_project
        # becomes the identity instead of acting along a noise direction
        self.assertAlmostEqual(float(load_out.numpy()[0]), 0.0, places=7)
        np.testing.assert_allclose(normal_out.numpy()[0], 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
