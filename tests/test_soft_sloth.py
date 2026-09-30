import unittest

import mujoco
import numpy as np

from mujoco_sim import prepare_mesh
from mujoco_sim.interactor import object_bounds, object_grip_lift_trajectory
from mujoco_sim.scene import ASSETS_DIR, load_model
from mujoco_sim.simulation import DigitalTwinSim

# Watertight sloth shape prepared from double_lift_sloth's shape prior, rolled
# onto its back and yawed so the head points at camera 0 (-y); the claw grips
# the torso, whose mid-body width still fits the open pad gap.
SLOTH_MESH = "object_sloth.stl"
# Coarse tetrahedra keep the smoke tests cheap; the assertions only need the
# sloth to be resolved enough to deform and be lifted.
SOFT = {"element_size": 0.05, "rgba": "0.47 0.42 0.37 1"}
# The palm of claw.xml sits this far above the claw's reference point.
PALM_HEIGHT = 0.148


def sloth_sim():
    model = load_model(1, None, SLOTH_MESH, SOFT)
    return DigitalTwinSim(model, width=80, height=60), model


def flex_vertices(sim, model, name="object"):
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    start = int(model.flex_vertadr[flex_id])
    count = int(model.flex_vertnum[flex_id])
    return np.array(sim.data.flexvert_xpos[start : start + count])


class SlothAssetTests(unittest.TestCase):
    def test_prepared_asset_is_closed_and_grippable(self):
        mesh = prepare_mesh.load_mesh(ASSETS_DIR / SLOTH_MESH)

        self.assertTrue(mesh.is_watertight)
        # Lies on its back: broad face down (the back plane rests on the floor),
        # thin along gravity.
        self.assertLess(float(mesh.extents[2]), 0.11)
        self.assertGreater(float(mesh.extents[1]), 0.30)
        # The claw closes along world x with a 0.20 m open gap between the
        # pads, so the section through the grip station (mid-body) has to fit.
        lo, hi = mesh.bounds
        section = mesh.section(
            plane_origin=[0.0, 0.5 * (lo[1] + hi[1]), 0.0], plane_normal=[0.0, 1.0, 0.0]
        )
        self.assertIsNotNone(section)
        grip_width = float(np.ptp(np.asarray(section.vertices)[:, 0]))
        self.assertLess(grip_width, 0.20)

    def test_sloth_compiles_as_a_soft_flex_wearing_its_texture(self):
        model = load_model(1, None, SLOTH_MESH, SOFT)

        self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_DISCRETE)
        flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "object")
        self.assertGreaterEqual(flex_id, 0)
        self.assertGreater(model.flex_vertnum[flex_id], 0)
        # The png/uvsrc sidecars port the source texture onto the flex via
        # MjSpec; the material replaces the manifest's tint, so the flex must
        # be white (rgba tints the texture) rather than the fallback colour.
        self.assertGreater(model.nflextexcoord, 0)
        self.assertGreaterEqual(int(model.flex_matid[flex_id]), 0)
        np.testing.assert_allclose(model.flex_rgba[flex_id], (1.0, 1.0, 1.0, 1.0))

    def test_grip_reaches_the_middle_without_the_palm_hitting_the_top(self):
        sim, model = sloth_sim()
        lower, upper = object_bounds(model, sim.data, "object")
        steps = 10
        trajectory, closing = object_grip_lift_trajectory(
            model, sim.data, steps_per_segment=steps
        )
        # Segment 1 is the hold at contact height.
        contact_z = trajectory[steps][0][2]

        self.assertGreaterEqual(contact_z + PALM_HEIGHT, upper[2] - 1e-9)
        self.assertGreater(contact_z, lower[2])
        self.assertLess(contact_z, upper[2])
        self.assertEqual(len(trajectory), 3 * steps)
        self.assertEqual(len(closing), len(trajectory))


class SlothGripLiftTests(unittest.TestCase):
    def test_grip_lift_raises_the_sloth_off_the_ground(self):
        sim, model = sloth_sim()
        trajectory, closing = object_grip_lift_trajectory(
            model, sim.data, steps_per_segment=60
        )
        sim.place_objects_on_ground()
        rest = flex_vertices(sim, model)

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
        final = flex_vertices(sim, model)

        self.assertGreater(len(frames), 2)
        self.assertGreater(float(final[:, 2].mean() - rest[:, 2].mean()), 0.05)
        self.assertGreater(float(final[:, 2].min()), 0.0)
        # The grasp tracks the claw, so the sloth lifts where it was grabbed
        # instead of drifting sideways.
        self.assertLess(abs(float(final[:, 0].mean() - rest[:, 0].mean())), 0.05)
        for frame in frames:
            self.assertTrue(np.isfinite(frame.object_qpos["object"]).all())


if __name__ == "__main__":
    unittest.main()
