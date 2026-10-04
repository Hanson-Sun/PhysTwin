"""Node mass must be physical, and the force scale must follow from it.

Masses used to be ``np.ones(n)`` -- 1.0 kg per node -- so an object weighed
``n * 9.81`` N (115 kN for the 11723-node seal). Node mass sets the force scale:
contact normal force is ``K * penetration`` and the Coulomb friction limit is
``mu * N``, both compared against the per-node weight ``m * g``. At 115 kN no
contact stiffness can hold the object, so friction grip is impossible.

These tests pin the total mass, the per-node spread, and the three stability
conditions that a low node mass implies:

    spring   omega = sqrt(Y/m)          -> Y * dt^2 / m must stay small
    contact  omega = sqrt(K/m)          -> K <=~ m * (0.5/dt)^2
    dashpot  stable iff dashpot*dt/m <~ 2

The first two are handled by mass-normalizing ``spring_Y``/``dashpot_damping``
inside the simulator, so the tests assert the SIMULATOR's effective values
rather than the raw config.
"""

import unittest

import numpy as np
import torch

from qqtt.utils import cfg
from qqtt.utils.mass_init import uniform_node_masses

DT = 5e-5
SEAL_NODES = 11723
SLOTH_NODES = 12741

# cfg.device is normally set by the training entry point (train_warp.py); this
# test constructs a simulator directly, so provide the same default.
if not hasattr(cfg, "device"):
    cfg.device = "cuda:0"


class UniformNodeMassTests(unittest.TestCase):
    def test_total_mass_is_the_configured_object_mass(self):
        for n in (SEAL_NODES, SLOTH_NODES):
            m = uniform_node_masses(n_object_nodes=n)
            self.assertAlmostEqual(
                float(m.sum()), cfg.object_total_mass, places=12,
                msg=f"n={n}: masses must sum to the object total mass",
            )

    def test_default_total_mass_matches_the_mujoco_soft_body(self):
        # mujoco_sim/soft_body.py DEFAULT_MASS = 0.3, written into the scene as
        # flexcomp mass="0.3". Keeping these equal means the sim reproduces the
        # weight of the object the ground truth came from.
        self.assertAlmostEqual(cfg.object_total_mass, 0.3, places=12)

    def test_mass_is_uniform_and_physical(self):
        m = uniform_node_masses(n_object_nodes=SEAL_NODES)
        self.assertEqual(m.shape, (SEAL_NODES,))
        self.assertTrue(np.allclose(m, m[0]))
        # the whole point: not 1.0 kg/node any more
        self.assertLess(m[0], 1e-4)
        self.assertGreater(m[0], 0.0)

    def test_controller_padding_is_excluded_from_the_total(self):
        """``points`` includes controller nodes, but they must not add mass.

        The simulator only reads ``masses[local_idx]`` for object-local indices,
        so mass placed on controller entries would be mass the object does not
        have to support.
        """
        n_ctrl = 30
        m = uniform_node_masses(n_object_nodes=SEAL_NODES,
                                total_length=SEAL_NODES + n_ctrl)
        self.assertEqual(m.shape, (SEAL_NODES + n_ctrl,))
        self.assertAlmostEqual(float(m[:SEAL_NODES].sum()),
                               cfg.object_total_mass, places=12)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            uniform_node_masses(n_object_nodes=0)
        with self.assertRaises(ValueError):
            uniform_node_masses(n_object_nodes=10, total_mass=0.0)
        with self.assertRaises(ValueError):
            uniform_node_masses(n_object_nodes=10, total_mass=-1.0)
        with self.assertRaises(ValueError):
            uniform_node_masses(n_object_nodes=10, total_mass=float("nan"))


class StabilityFollowsMassTests(unittest.TestCase):
    """The force coefficients must be interpreted against the new node mass."""

    def setUp(self):
        from qqtt.model.diff_simulator import spring_mass_warp as smw
        self.smw = smw

    def _build(self, n_nodes, ctrl_points, spring_Y, dashpot):
        """Build a simulator over a trivial 2-node chain to read its coefficients."""
        base_springs = torch.tensor([[0, 1]], dtype=torch.int32, device=cfg.device)
        rest = torch.tensor([0.01], dtype=torch.float32, device=cfg.device)
        masses = uniform_node_masses(n_nodes)
        sim = self.smw.SpringMassSystemWarp(
            base_springs=base_springs,
            base_rest_lengths=rest,
            init_masses=torch.tensor(masses, dtype=torch.float32, device=cfg.device),
            init_masks=None,
            signed_incidence_map=None,
            max_incident_springs=0,
            init_vertices=torch.zeros(n_nodes, 3, dtype=torch.float32, device=cfg.device),
            init_velocities=torch.zeros(n_nodes, 3, dtype=torch.float32, device=cfg.device),
            dt=DT,
            num_substeps=2,
            dashpot_damping=dashpot,
            drag_damping=cfg.drag_damping,
            collision_dist=cfg.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            self_collision=False,
            collide_elas=torch.tensor([cfg.collide_elas], device=cfg.device),
            collide_fric=torch.tensor([cfg.collide_fric], device=cfg.device),
            collide_object_elas=torch.tensor([cfg.collide_object_elas], device=cfg.device),
            collide_object_fric=torch.tensor([cfg.collide_object_fric], device=cfg.device),
            spring_Y=torch.full((1,), spring_Y, dtype=torch.float32, device=cfg.device),
            object_massnodes_total=n_nodes,
            object_massnodes_single=n_nodes,
            controller_massnodes_single=len(ctrl_points),
            controller_rest_location=torch.tensor(
                ctrl_points, dtype=torch.float32, device=cfg.device),
            number_of_instance=1,
        )
        return sim

    def test_dashpot_is_mass_normalized(self):
        """Damping stability needs dashpot*dt/m < 2; unnormalized it was 195."""
        dashpot = cfg.dashpot_damping
        n = SEAL_NODES
        ctrl = [[1.0, 0.0, 0.0]]
        sim = self._build(n, ctrl, cfg.init_spring_Y, dashpot)
        m = cfg.object_total_mass / n
        ratio = sim.dashpot_damping * DT / m
        self.assertLess(
            ratio, 1.0,
            f"dashpot*dt/m = {ratio:.3g} is unstable (must be <~2)",
        )
        # and it equals the config value per unit mass. dashpot_damping is stored as
        # a Python float but derived from a float32 mass mean, so it carries
        # ~7 significant digits, not float64. Compare relatively at 1e-6.
        self.assertLess(
            abs(sim.dashpot_damping / dashpot - m) / m, 1e-6,
            "dashpot must be the config value scaled by the node mass",
        )

    def test_spring_stiffness_is_mass_normalized(self):
        """omega*dt = sqrt(Y/m)*dt*sqrt(lambda/rest); stays ~unchanged."""
        n = SEAL_NODES
        ctrl = [[1.0, 0.0, 0.0]]
        sim = self._build(n, ctrl, cfg.init_spring_Y, cfg.dashpot_damping)
        m = cfg.object_total_mass / n
        Y_eff = float(
            torch.as_tensor(sim.wp_spring_Y_clamped.numpy()).float().mean()
        )
        # compare omega*dt before/after normalization at a representative mode
        lam_over_rest = 51.687 / 0.010991  # measured on the real seal graph
        w_old = np.sqrt(cfg.init_spring_Y * lam_over_rest / 1.0) * DT
        w_new = np.sqrt(Y_eff * lam_over_rest / m) * DT
        self.assertAlmostEqual(
            w_new / w_old, 1.0, places=3,
            msg="mass-normalizing spring_Y must leave omega*dt unchanged",
        )
        self.assertLess(w_new, 1.0)

    def test_contact_stiffness_within_explicit_integration_limit(self):
        """Contact is absolute (not normalized), so it must respect K <= m(c/dt)^2.

        Left at the old 3e4 this is ~12x over the limit and the sim NaNs.
        """
        for n in (SEAL_NODES, SLOTH_NODES):
            m = cfg.object_total_mass / n
            k_max = m * (0.5 / DT) ** 2
            self.assertLessEqual(
                cfg.controller_contact_stiffness, k_max,
                f"n={n}: contact stiffness must stay under {k_max:.1f} N/m",
            )

    def test_contact_stiffness_can_still_hold_the_object(self):
        """Lowering K for stability must not destroy grip.

        Friction can hold the object when mu*K*sum_penetration >= weight. Using
        the summed penetration measured on the real seal grip at frame 45.
        """
        sum_penetration = 0.902  # metres, dense collider, real recorded frame
        for n in (SEAL_NODES, SLOTH_NODES):
            margin = (
                cfg.controller_contact_friction
                * cfg.controller_contact_stiffness
                * sum_penetration
                / (cfg.object_total_mass * 9.81)
            )
            self.assertGreater(
                margin, 1.0,
                f"n={n}: friction margin {margin:.3f}x is too small to lift",
            )

    def test_simulator_rejects_unstable_contact_stiffness(self):
        """An over-limit K must fail loudly, not NaN mid-episode."""
        n = 200
        ctrl = [[1.0, 0.0, 0.0]]
        original = cfg.controller_contact_stiffness
        cfg.controller_contact_stiffness = 1e9
        try:
            with self.assertRaises(ValueError) as ctx:
                self._build(n, ctrl, cfg.init_spring_Y, cfg.dashpot_damping)
            self.assertIn("controller_contact_stiffness", str(ctx.exception))
        finally:
            cfg.controller_contact_stiffness = original

    def test_stability_guard_bounds_k_by_scaled_mass_not_ref_mass(self):
        """K is bounded by the mass in wp_masses (ref_mass*init_mass), not ref_mass.

        CMA routinely sets init_mass well below 1, so a guard built on ref_mass
        over-permits K by exactly 1/init_mass. On sim_soft_ball_grip_lift
        (init_mass=0.150) that let K=2000 through a check whose real limit was
        897 N/m. Only spring_mass_warp_upstream scales masses by init_mass, so
        that is the module under test here.
        """
        from qqtt.model.diff_simulator import spring_mass_warp_upstream as smwu

        n = 5027  # sim_soft_ball_grip_lift object node count
        original_mass = cfg.init_mass
        original_k = cfg.controller_contact_stiffness
        original_type = cfg.data_type
        original_graph = cfg.use_graph
        cfg.init_mass = 0.150
        cfg.data_type = "sim"  # skips the gt-visibility arrays the guard does not touch
        cfg.use_graph = False  # graph capture needs a full scene; guard runs earlier
        try:
            ref_mass = cfg.object_total_mass / n
            true_limit = ref_mass * cfg.init_mass * (0.5 / DT) ** 2
            stale_limit = ref_mass * (0.5 / DT) ** 2

            def build():
                masses = torch.tensor(
                    uniform_node_masses(n), dtype=torch.float32, device=cfg.device
                )
                return smwu.SpringMassSystemWarp(
                    init_vertices=torch.zeros(n, 3, dtype=torch.float32, device=cfg.device),
                    init_springs=torch.tensor([[0, 1]], dtype=torch.int32, device=cfg.device),
                    init_rest_lengths=torch.tensor([0.01], dtype=torch.float32, device=cfg.device),
                    init_masses=masses,
                    dt=DT,
                    num_substeps=2,
                    spring_Y=torch.tensor([cfg.init_spring_Y], dtype=torch.float32, device=cfg.device),
                    collide_elas=torch.tensor([cfg.collide_elas], device=cfg.device),
                    collide_fric=torch.tensor([cfg.collide_fric], device=cfg.device),
                    dashpot_damping=cfg.dashpot_damping,
                    drag_damping=cfg.drag_damping,
                    num_object_points=n,
                    gt_object_points=torch.zeros(
                        3, n, 3, dtype=torch.float32, device=cfg.device
                    ),
                )

            # Legal under the stale ref_mass bound, illegal under the real one.
            cfg.controller_contact_stiffness = 0.5 * (stale_limit + true_limit)
            with self.assertRaises(ValueError) as ctx:
                build()
            self.assertIn("controller_contact_stiffness", str(ctx.exception))

            # And a value under the real limit must still be accepted.
            cfg.controller_contact_stiffness = 0.5 * true_limit
            build()
        finally:
            cfg.init_mass = original_mass
            cfg.controller_contact_stiffness = original_k
            cfg.data_type = original_type
            cfg.use_graph = original_graph


if __name__ == "__main__":
    unittest.main()