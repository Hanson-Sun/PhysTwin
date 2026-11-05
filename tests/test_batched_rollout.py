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

from asid.fim.case import load_case
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


if __name__ == "__main__":
    unittest.main()
