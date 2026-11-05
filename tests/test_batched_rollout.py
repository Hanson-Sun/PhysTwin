"""Batched simulator and rollout equivalence.

The batched simulator holds instance-major parameters: ``k * N`` masses,
``k * n_springs`` stiffnesses and ``k`` dashpots. These tests pin the tiling and
check that identical instances produce identical dynamics before the rollout
harness is built on top.
"""

import unittest
from pathlib import Path

import numpy as np
import torch
import warp as wp

from asid.fim.batched import BatchedRollout
from asid.fim.case import load_case
from asid.fim.fim import Fisher, Prior
from asid.fim.group import build_partition
from asid.fim.operators import (
    default_observation,
    jacobian,
    jacobian_batched,
    parameterized_run,
)
from asid.fim.runner import BatchedActionRunner
from asid.fim.simulator import Rollout, SimConfig
from qqtt.model.diff_simulator import spring_mass_warp as smw
from qqtt.utils.mass_init import uniform_node_masses

CASE_DIR = Path("data/different_types/sim_soft_sloth_push")
DEVICE = "cuda:0"


def tile(values: np.ndarray, instances: int) -> np.ndarray:
    return np.tile(values, (instances,) + (1,) * (values.ndim - 1))


@unittest.skipUnless((CASE_DIR / "final_data.pkl").is_file(), "case data not available")
class SimulatorBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qqtt.utils import cfg

        cfg.device = DEVICE
        cfg.data_type = "real"
        cfg.reverse_z = True
        cls.case = load_case(CASE_DIR, num_nodes=120, max_frames=6)
        cls.masses = uniform_node_masses(len(cls.case.nodes), total_mass=0.3)

    def build(self, instances: int, batched_forces: bool):
        nodes = len(self.case.nodes)
        controllers = len(self.case.controller[0])
        vertices = np.concatenate(
            [tile(self.case.nodes, instances), tile(self.case.controller[0], instances)]
        )
        return smw.SpringMassSystemWarp(
            base_springs=torch.tensor(
                self.case.springs, dtype=torch.int32, device=DEVICE
            ),
            base_rest_lengths=torch.tensor(
                self.case.rest_lengths, dtype=torch.float32, device=DEVICE
            ),
            init_masses=torch.tensor(self.masses, device=DEVICE),
            init_masks=None,
            signed_incidence_map=None,
            max_incident_springs=0,
            init_vertices=torch.tensor(vertices, device=DEVICE),
            init_velocities=None,
            dt=1.0 / 30.0 / 100.0,
            num_substeps=100,
            dashpot_damping=100.0,
            drag_damping=3.0,
            collision_dist=0.02,
            reverse_z=True,
            spring_Y_min=0.0,
            spring_Y_max=1e5,
            self_collision=False,
            collide_elas=torch.tensor(0.5, device=DEVICE),
            collide_fric=torch.tensor(0.5, device=DEVICE),
            collide_object_elas=torch.tensor(0.7, device=DEVICE),
            collide_object_fric=torch.tensor(0.3, device=DEVICE),
            spring_Y=torch.tensor(3e4, device=DEVICE),
            object_massnodes_total=instances * nodes,
            object_massnodes_single=nodes,
            controller_massnodes_single=controllers,
            controller_rest_location=torch.tensor(
                tile(self.case.controller[0], instances), device=DEVICE
            ),
            number_of_instance=instances,
            controller_contact_friction=0.3,
            sim_force_mode=(
                smw.SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC
                if batched_forces
                else smw.SIM_FORCE_MODE_GATHER
            ),
        )

    def drive(self, sim, frames: int = 2):
        tiled = np.tile(
            self.case.controller[: frames + 1], (1, sim.number_of_instance, 1)
        )
        for frame in range(1, frames + 1):
            last = np.ascontiguousarray(tiled[frame - 1])
            target = np.ascontiguousarray(tiled[frame])
            sim.set_controller_interactive(
                wp.from_torch(torch.tensor(last, device=DEVICE), dtype=wp.vec3),
                wp.from_torch(torch.tensor(target, device=DEVICE), dtype=wp.vec3),
            )
            sim.step()
        return wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).cpu()

    def test_shared_parameters_are_tiled_not_reinterpreted(self):
        sim = self.build(3, batched_forces=True)
        nodes = len(self.case.nodes)
        masses = wp.to_torch(sim.wp_masses, requires_grad=False).cpu()
        np.testing.assert_allclose(masses[:nodes], self.masses, rtol=1e-6)
        np.testing.assert_allclose(masses.reshape(3, nodes)[1], self.masses, rtol=1e-6)
        self.assertEqual(sim.wp_spring_Y_clamped.shape[0], 3 * sim.n_springs_single)
        self.assertEqual(sim.wp_dashpot_damping.shape[0], 3)

    def test_identical_instances_step_identically(self):
        single = self.drive(self.build(1, batched_forces=False))
        batched = self.drive(self.build(3, batched_forces=True))
        nodes = len(self.case.nodes)
        for index in range(3):
            difference = (batched[index * nodes : (index + 1) * nodes] - single).abs().max()
            self.assertLess(float(difference), 1e-5, f"instance {index} diverged")


@unittest.skipUnless((CASE_DIR / "final_data.pkl").is_file(), "case data not available")
class BatchedRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qqtt.utils import cfg

        cfg.device = DEVICE
        cfg.data_type = "real"
        cfg.reverse_z = True
        cls.case = load_case(CASE_DIR, num_nodes=200, max_frames=8)
        cls.config = SimConfig(frames=6, num_substeps=100)
        cls.partition = build_partition(cls.case, groups=2, config=cls.config)

    def test_single_instance_matches_the_serial_rollout(self):
        serial = Rollout(self.case, self.config).run()
        batched = BatchedRollout(self.case, self.config, 1).run()
        difference = float((batched - serial).abs().max())
        print(f"k=1 vs serial: max|d| = {difference:.3e}")
        # The serial contact query widens by the controller sweep radius; the
        # candidate set differs, the accepted contacts do not.
        self.assertLess(difference, 1e-5)

    def test_per_instance_parameters_match_individual_runs(self):
        rows = self.partition.phi0 + np.random.default_rng(0).normal(
            scale=0.05, size=(3, self.partition.dimension)
        )
        batch = BatchedRollout(self.case, self.config, 3)
        batch.set_parameters(rows, self.partition)
        together = batch.run()
        nodes = len(self.case.nodes)
        for index, row in enumerate(rows):
            single = BatchedRollout(self.case, self.config, 1)
            single.set_parameters(row[None, :], self.partition)
            difference = (
                together[:, index * nodes : (index + 1) * nodes] - single.run()
            ).abs().max()
            self.assertLess(float(difference), 1e-6, f"instance {index} diverged")

    def test_batched_jacobian_matches_within_difference_noise(self):
        """The reference has its own finite-difference noise floor.

        A 1e-5 relative change of the step moves the serial J by 0.27% here
        (9.5% at 12 frames), because ground friction and contact switch
        discontinuously. Compare the batched J against that measured floor
        instead of an absolute tolerance: a genuine bug is much larger.
        """
        observation = default_observation(self.config.frames, stride=2)
        prior = Prior().std(self.partition.groups)
        serial = parameterized_run(
            Rollout(self.case, self.config), self.partition, observation
        )
        step = 1e-2
        exact = jacobian(serial, self.partition.phi0, step, central=True)
        shifted = jacobian(
            serial, self.partition.phi0, step * (1.0 + 1e-5), central=True
        )
        noise = float(
            np.abs(exact - shifted).max() / np.abs(exact).max()
        )
        runner = BatchedActionRunner(self.case, self.config)
        batched_run = runner.run(
            self.case.controller, self.case.contact, self.partition, observation
        )
        approx = jacobian_batched(batched_run, self.partition.phi0, step)
        relative = float(
            np.abs(approx - exact).max() / np.abs(exact).max()
        )
        exact_gain = Fisher.from_jacobian(exact, observation.sigma, prior).logdet()
        approx_gain = Fisher.from_jacobian(approx, observation.sigma, prior).logdet()
        gain_relative = abs(approx_gain - exact_gain) / abs(exact_gain)
        print(
            f"batched J: rel {relative:.3e} (serial step noise {noise:.3e}), "
            f"logdet rel {gain_relative:.3e} "
            f"({exact_gain:.6f} vs {approx_gain:.6f})"
        )
        self.assertLess(relative, max(1e-3, 2.0 * noise))
        self.assertLess(gain_relative, 1e-2)


if __name__ == "__main__":
    unittest.main()
