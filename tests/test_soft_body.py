import unittest

import mujoco
import numpy as np

from mujoco_sim import soft_body
from mujoco_sim.interactor import (
    object_bounds,
    object_grip_lift_trajectory,
)
from mujoco_sim.scene import ASSETS_DIR, load_model
from mujoco_sim.simulation import DigitalTwinSim


BALL_MESH = "ball.obj"
# Coarse tetrahedra keep the smoke tests cheap; the assertions only need the
# ball to be resolved enough to deform.
SOFT = {"element_size": 0.03}


def flex_layout(model, name="object"):
    """Return (flex_id, first vertex, vertex count) for a named flex."""
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    if flex_id < 0:
        raise AssertionError(f"model has no flex named '{name}'")
    start = int(model.flex_vertadr[flex_id])
    return flex_id, start, int(model.flex_vertnum[flex_id])


def flex_vertices(sim, start, count):
    return np.array(sim.data.flexvert_xpos[start : start + count])


def soft_sim():
    model = load_model(1, None, BALL_MESH, SOFT)
    return DigitalTwinSim(model, width=80, height=60), model


class TetrahedralizeTests(unittest.TestCase):
    def test_arbitrary_mesh_becomes_tetrahedra(self):
        nodes, tets = soft_body.tetrahedralize(ASSETS_DIR / BALL_MESH, 0.03)

        self.assertEqual(nodes.shape[1], 3)
        self.assertEqual(tets.shape[1], 4)
        self.assertGreater(len(tets), 0)
        # Tetrahedra index the node array, so tags must be rebased to 0..n-1.
        self.assertGreaterEqual(int(tets.min()), 0)
        self.assertLess(int(tets.max()), len(nodes))

    def test_missing_or_unsupported_input_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            soft_body.tetrahedralize(ASSETS_DIR / "does_not_exist.obj")
        with self.assertRaises(ValueError):
            soft_body.tetrahedralize(ASSETS_DIR / "world.xml")

    def test_serialised_mesh_is_a_single_node_block(self):
        nodes, tets = soft_body.tetrahedralize(ASSETS_DIR / BALL_MESH, 0.03)
        text = soft_body.gmsh_bytes(nodes, tets).decode()

        # MuJoCo rejects GMSH files that split nodes across entity blocks.
        self.assertEqual(text.count("$Nodes"), 1)
        self.assertIn(f"1 {len(nodes)} 1 {len(nodes)}", text)



class SoftModelTests(unittest.TestCase):
    def test_soft_object_is_a_named_flex_on_the_discrete_integrator(self):
        model = load_model(1, None, BALL_MESH, SOFT)

        # MuJoCo ignores flex elasticity under the scene's implicitfast
        # integrator, which is why the soft asset has to select discrete.
        self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_DISCRETE)
        _, start, count = flex_layout(model)
        self.assertGreater(count, 0)
        # The flexcomp stays under the usual object body, so trajectories and
        # bounds lookups keep addressing the object by name.
        self.assertGreaterEqual(model.body("object").id, 0)
        self.assertGreaterEqual(start, 0)

    def test_soft_object_takes_the_configured_colour(self):
        model = load_model(1, None, BALL_MESH, SOFT)

        # Defaults to a blue that no other body in the scene shares, so
        # colour-prompted masking cannot latch onto the claw's orange spheres.
        expected = [float(v) for v in soft_body.DEFAULT_RGBA.split()]
        np.testing.assert_allclose(model.flex_rgba[0], expected)

        custom = load_model(1, None, BALL_MESH, {**SOFT, "rgba": "0 0 1 1"})
        np.testing.assert_allclose(custom.flex_rgba[0], (0.0, 0.0, 1.0, 1.0))

    def test_soft_mesh_starts_resting_on_the_ground(self):
        sim, model = soft_sim()
        _, start, count = flex_layout(model)

        z = flex_vertices(sim, start, count)[:, 2]
        # A flex has no freejoint, so place_objects_on_ground cannot move it;
        # the asset itself bakes the ground placement in. Vertices collide as
        # spheres of `radius`, so the rest pose sits that far above the floor:
        # grounding the vertices would embed them in the floor and make the
        # solver push the body upward on the first steps.
        self.assertAlmostEqual(float(z.min()), soft_body.DEFAULT_RADIUS, places=6)

    def test_get_object_state_reports_the_flex_centroid(self):
        sim, model = soft_sim()
        _, start, count = flex_layout(model)

        state = sim.get_object_state("object")
        self.assertEqual(state.shape, (7,))
        np.testing.assert_allclose(state[:3], flex_vertices(sim, start, count).mean(0))
        np.testing.assert_allclose(state[3:], (1.0, 0.0, 0.0, 0.0))

    def test_rigid_object_state_is_unchanged(self):
        model = load_model(1, "object_box.xml")
        sim = DigitalTwinSim(model, width=80, height=60)

        np.testing.assert_allclose(
            sim.get_object_state("object"), sim.get_free_body_state("object")
        )

    def test_translate_flex_moves_the_whole_mesh(self):
        sim, model = soft_sim()
        flex_id, start, count = flex_layout(model)
        rest = flex_vertices(sim, start, count)
        delta = np.array([0.04, 0.0, 0.02])

        sim.translate_flex(flex_id, delta)
        mujoco.mj_forward(model, sim.data)
        moved = flex_vertices(sim, start, count)

        np.testing.assert_allclose(moved.mean(0) - rest.mean(0), delta, atol=1e-5)
        # A rigid translation must not change the mesh's shape.
        np.testing.assert_allclose(
            moved - moved.mean(0), rest - rest.mean(0), atol=1e-6
        )


class SoftTrajectoryTests(unittest.TestCase):
    """End-to-end smoke tests: the claw has to really touch the flex."""

    def test_grip_contact_height_is_the_centre_for_short_objects(self):
        sim, model = soft_sim()
        lower, upper = object_bounds(model, sim.data, "object")
        trajectory, _ = object_grip_lift_trajectory(
            model, sim.data, steps_per_segment=10, end_pause_seconds=0.05
        )

        # The palm-clearance rule only applies to objects taller than the
        # palm's reach, so a ball keeps being gripped at its centre.
        self.assertAlmostEqual(
            trajectory[10][0][2], float((lower[2] + upper[2]) * 0.5), places=6
        )

    def test_grip_lift_raises_the_soft_ball_vertically(self):
        sim, model = soft_sim()
        _, start, count = flex_layout(model)
        trajectory, closing = object_grip_lift_trajectory(
            model, sim.data, steps_per_segment=30, end_pause_seconds=0.05
        )
        sim.place_objects_on_ground()
        rest = flex_vertices(sim, start, count)

        frames = sim.rollout(
            {"interactor0": trajectory},
            object_bodies=["object"],
            capture_every=10,
            substeps=4,
            gripper_opening={"interactor0": closing},
            grasped_body="object",
            grasp_offset=(0.0, 0.0, 0.0),
            max_controller_speed=0.25,
            controller_free_speed=0.25,
            controller_slowdown_epsilon=0.02,
        )
        final = flex_vertices(sim, start, count)

        self.assertGreater(len(frames), 2)
        lifted = final[:, 2].mean() - rest[:, 2].mean()
        self.assertGreater(float(lifted), 0.05)
        # The grasp only tracks the claw, so the lift stays vertical.
        self.assertLess(abs(float(final[:, 0].mean() - rest[:, 0].mean())), 0.01)
        self.assertGreater(float(final[:, 2].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
