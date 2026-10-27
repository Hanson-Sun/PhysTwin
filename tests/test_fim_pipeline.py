"""End-to-end parameter map, Jacobian, Fisher, and sketch on a small mesh."""

import unittest

import numpy as np

from asid.fim.case import Case, build_object_springs
from asid.fim.fim import Fisher, Prior
from asid.fim.group import build_partition
from asid.fim.operators import Observation, jacobian, parameterized_run
from asid.fim.simulator import Rollout, SimConfig
from asid.fim.sketch import randomized_sketch


def synthetic_case(frames: int = 4) -> Case:
    axis = np.arange(4) * 0.05
    nodes = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)
    springs, rest = build_object_springs(nodes, radius=0.08, max_neighbours=12)
    controller = np.zeros((frames, 1, 3), dtype=np.float32)
    controller[:, 0, 0] = np.linspace(0.0, 0.1, frames)
    controller[:, 0, 2] = 0.12
    return Case(
        name="synthetic",
        nodes=nodes.astype(np.float32),
        springs=springs,
        rest_lengths=rest,
        num_object_springs=len(springs),
        controller=controller,
        contact=controller.copy(),
    )


def relative_change(coarse: np.ndarray, fine: np.ndarray) -> float:
    """Relative Frobenius distance between two Jacobian estimates."""
    return float(np.linalg.norm(coarse - fine) / np.linalg.norm(fine))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = synthetic_case()
        cls.config = SimConfig(frames=4, num_substeps=100)
        cls.partition = build_partition(
            cls.case, groups=3, seed=1, config=cls.config
        )
        cls.observation = Observation(frames=np.array([0, 2, 3]), sigma=0.005)
        cls.rollout = Rollout(cls.case, cls.config)
        cls.evaluate = staticmethod(
            parameterized_run(cls.rollout, cls.partition, cls.observation)
        )
        cls.phi0 = cls.partition.phi0

    def test_phi0_matches_the_nominal_simulator(self):
        reference = Observation(frames=self.observation.frames)
        nominal = reference.vector(Rollout(self.case, self.config).run())
        np.testing.assert_allclose(self.evaluate(self.phi0), nominal, atol=1e-6)

    def test_jacobian_is_finite_and_active(self):
        jac = jacobian(
            self.evaluate, self.phi0, step=1e-2, indices=[0, 3, 6], central=False
        )
        self.assertEqual(jac.shape[0], 3)
        self.assertTrue(np.isfinite(jac).all())
        self.assertGreater(np.abs(jac).max(), 0.0)

    def test_exact_fisher_is_psd_and_sketch_is_a_lower_bound(self):
        jac = jacobian(self.evaluate, self.phi0, step=1e-2, central=False)
        prior = Prior().std(self.partition.groups)
        fisher = Fisher.from_jacobian(jac, self.observation.sigma, prior)
        self.assertGreaterEqual(fisher.eigenvalues().min(), -1e-8)
        sketch = randomized_sketch(
            self.evaluate,
            self.phi0,
            prior,
            self.observation.sigma,
            rank=3,
            step=1e-2,
            seed=2,
        )
        self.assertLessEqual(sketch.gain, fisher.logdet() + 1e-6)
        self.assertTrue(np.isfinite(sketch.U).all())


class JacobianAccuracyTests(unittest.TestCase):
    """How accurate the finite-difference Jacobian is at the default step.

    Every Fisher in this package is ``J J^T / sigma^2``, so a systematic error
    in ``J`` propagates straight into the information map; the scripts default
    to ``--step 1e-2`` and this pins what that buys. The trajectory is only
    piecewise smooth in ``phi``: the contact penalty turns on over a finite
    band, so the difference quotient moves with the step until the step is
    small enough to stay inside one activation pattern.

    Accuracy is bounded by halving the step, not by comparing against a known
    derivative - the harness runs the simulator at ``requires_grad=False``, so
    there is no autodiff reference to check against.
    """

    FRAMES = 6

    @classmethod
    def setUpClass(cls):
        cls.case = synthetic_case(frames=cls.FRAMES)
        config = SimConfig(frames=cls.FRAMES, num_substeps=100)
        cls.partition = build_partition(cls.case, groups=3, seed=1, config=config)
        cls.observation = Observation(frames=np.array([0, 2, 4, 5]), sigma=0.005)
        # One spring group, the dashpot and one mass group: every block of phi,
        # without paying for all 2g + 1 columns.
        cls.indices = [0, 3, 6]

    def jacobians(self, steps, soft_ground: bool = True):
        config = SimConfig(
            frames=self.FRAMES, num_substeps=100, soft_ground=soft_ground
        )
        run = parameterized_run(
            Rollout(self.case, config), self.partition, self.observation
        )
        return {
            step: jacobian(
                run,
                self.partition.phi0,
                step=step,
                indices=self.indices,
                central=True,
            )
            for step in steps
        }

    def test_the_default_step_is_converged(self):
        js = self.jacobians((1e-2, 5e-3))
        change = relative_change(js[1e-2], js[5e-3])
        self.assertLess(
            change,
            0.15,
            msg=f"halving the default 1e-2 step moved J by {change:.1%}; the "
            "quotient is still dominated by the step, not by the model",
        )

    def test_hard_contact_would_not_converge(self):
        # The reason the harness smooths ground contact: an impulse makes the
        # contact set jump as phi moves, so no step gives a stable quotient.
        js = self.jacobians((1e-2, 5e-3), soft_ground=False)
        change = relative_change(js[1e-2], js[5e-3])
        self.assertGreater(
            change,
            0.25,
            msg=f"the hard-contact quotient moved by only {change:.1%}: if the "
            "impulse has become smooth, revisit the penalty surrogate rather "
            "than raising this bound",
        )


if __name__ == "__main__":
    unittest.main()
