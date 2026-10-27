"""Partitioning and phi/theta maps."""

import unittest
from pathlib import Path

import numpy as np

from asid.fim.case import Case, build_object_springs
from asid.fim.group import build_partition, kmeans
from asid.fim.simulator import SimConfig

CASE_DIR = Path("data/different_types/sim_soft_sloth_push")


class KMeansTests(unittest.TestCase):
    def test_deterministic_and_all_groups_used(self):
        points = np.random.default_rng(0).normal(size=(200, 3))
        first, centers = kmeans(points, 7, seed=3)
        second, other = kmeans(points, 7, seed=3)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(centers, other)
        self.assertEqual(len(np.unique(first)), 7)

    def test_three_blobs_are_recovered(self):
        centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        points = np.concatenate(
            [center + 0.01 * np.random.default_rng(i).normal(size=(30, 3))
             for i, center in enumerate(centers)]
        )
        labels, _ = kmeans(points, 3, seed=0)
        for block in range(3):
            self.assertEqual(
                len(np.unique(labels[block * 30 : (block + 1) * 30])), 1
            )
        self.assertEqual(len(np.unique(labels)), 3)


def synthetic_case() -> Case:
    axis = np.arange(4) * 0.05
    nodes = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)
    springs, rest = build_object_springs(nodes, radius=0.08, max_neighbours=12)
    controller = np.zeros((2, 1, 3), dtype=np.float32)
    controller[0, 0] = [0.0, 0.0, 0.1]
    controller[1, 0] = [0.1, 0.0, 0.1]
    return Case(
        name="synthetic",
        nodes=nodes.astype(np.float32),
        springs=springs,
        rest_lengths=rest,
        num_object_springs=len(springs),
        controller=controller,
        contact=controller.copy(),
    )


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.case = synthetic_case()
        self.config = SimConfig(frames=2)
        self.partition = build_partition(self.case, groups=4, seed=1, config=self.config)

    def test_dimension_and_indices(self):
        self.assertEqual(self.partition.dimension, 9)
        self.assertEqual(len(self.partition.phi0), 9)
        self.assertEqual(self.partition.node_groups.shape, (len(self.case.nodes),))
        self.assertEqual(
            self.partition.spring_groups.shape, (self.case.num_object_springs,)
        )
        self.assertEqual(len(np.unique(self.partition.node_groups)), 4)

    def test_phi0_round_trips_to_nominal(self):
        phi = self.partition.phi0
        np.testing.assert_allclose(
            self.partition.spring_Y(phi), self.config.spring_Y
        )
        np.testing.assert_allclose(
            self.partition.masses(phi),
            np.full(len(self.case.nodes), self.config.total_mass / len(self.case.nodes)),
        )
        self.assertAlmostEqual(self.partition.dashpot(phi), self.config.dashpot_damping)

    def test_group_perturbation_only_touches_its_members(self):
        phi = self.partition.phi0.copy()
        phi[0] += 0.5
        values = self.partition.spring_Y(phi)
        affected = np.flatnonzero(
            ~np.isclose(values, self.config.spring_Y, rtol=1e-9, atol=0.0)
        )
        np.testing.assert_array_equal(
            np.unique(self.partition.spring_groups[affected]), [0]
        )


@unittest.skipUnless((CASE_DIR / "final_data.pkl").is_file(), "case data not available")
class RealPartitionTests(unittest.TestCase):
    def test_real_case_groups(self):
        from asid.fim.case import load_case

        case = load_case(CASE_DIR, num_nodes=500, max_frames=4)
        partition = build_partition(case, groups=10, seed=0)
        self.assertEqual(partition.node_groups.shape, (500,))
        self.assertEqual(partition.spring_groups.shape, (case.num_object_springs,))
        self.assertEqual(partition.dimension, 21)
        self.assertEqual(len(np.unique(partition.node_groups)), 10)
        self.assertTrue(np.all(partition.spring_groups >= 0))
        self.assertTrue(np.all(partition.spring_groups < 10))


if __name__ == "__main__":
    unittest.main()
