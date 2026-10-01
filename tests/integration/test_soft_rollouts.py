"""Full-pipeline soft-body tests: long rollouts and real-asset repair.

Run with ``python -m unittest tests.integration``. These take tens of seconds
each; the quick smoke suite lives at ``tests``.
"""

import unittest

import numpy as np

from mujoco_sim import soft_body
from mujoco_sim.interactor import object_push_trajectory
from mujoco_sim.scene import ASSETS_DIR
from tests.test_soft_body import flex_layout, flex_vertices, soft_sim


class SoftRolloutTests(unittest.TestCase):
    def test_push_translates_and_deforms_the_soft_ball(self):
        sim, model = soft_sim()
        _, start, count = flex_layout(model)
        trajectory = object_push_trajectory(
            model, sim.data, steps_per_segment=40, end_pause_seconds=0.05
        )
        sim.place_objects_on_ground()
        rest = flex_vertices(sim, start, count)

        frames = sim.rollout(
            {"interactor0": trajectory},
            object_bodies=["object"],
            capture_every=5,
            substeps=4,
            max_controller_speed=0.2,
            controller_free_speed=0.7,
            controller_slowdown_epsilon=0.02,
        )
        final = flex_vertices(sim, start, count)

        self.assertGreater(len(frames), 2)
        self.assertGreater(float(final[:, 0].mean() - rest[:, 0].mean()), 0.005)
        # The claw squashes the ball on contact, so the shape must change.
        shape = final - final.mean(0) - (rest - rest.mean(0))
        self.assertGreater(float(np.linalg.norm(shape, axis=1).max()), 0.001)
        for frame in frames:
            self.assertTrue(np.isfinite(frame.object_qpos["object"]).all())


class VoxelRepairTests(unittest.TestCase):
    def test_closed_surface_that_gmsh_cannot_reparametrise_still_fills(self):
        # A voxel-repaired surface (what prepare_mesh.watertight writes) is
        # already manifold and has no seam for gmsh's geometry repair to snap,
        # so tetrahedralize has to fall back to using its facets directly.
        nodes, tets = soft_body.tetrahedralize(ASSETS_DIR / "object_sloth.stl", 0.05)

        self.assertGreater(len(tets), 0)
        # Tetrahedra index the node array, so tags must be rebased to 0..n-1.
        self.assertGreaterEqual(int(tets.min()), 0)
        self.assertLess(int(tets.max()), len(nodes))
