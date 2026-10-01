"""Physical grip must be friction only: no hidden claw-to-object attachment.

``mujoco_sim.grasp`` offers two ways to carry an object once the fingers close
- a ``weld`` equality for rigid bodies and a driven flex root frame for soft
ones. Both are legitimate, but only for ``grasp_mode="constraint"``. In
``physical`` mode the object must hang from pad friction alone, or the dataset
claims a friction grip it never had.

These tests check the invariant from both sides: physical engages nothing, and
constraint still does (so the check above cannot pass vacuously).
"""
import unittest

import mujoco
import numpy as np

from mujoco_sim.grasp import flex_root_body
from mujoco_sim.interactor import object_grip_lift_trajectory
from mujoco_sim.scene import load_model
from mujoco_sim.simulation import DigitalTwinSim

BALL_MESH = "ball.obj"
SOFT = {"element_size": 0.03}
WELD = "interactor0_grasp_weld"


class AttachAudit(DigitalTwinSim):
    """DigitalTwinSim that records whether any attach is ever engaged."""

    def __post_init__(self):
        super().__post_init__()
        self.weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD
        )
        self.root_body = (
            flex_root_body(self.model, 0) if self.model.nflexvert else -1
        )
        self.weld_ever_active = False
        self.root_ever_driven = False

    def step(self, n, motion=None):
        super().step(n, motion=motion)
        if self.weld_id >= 0 and bool(self.data.eq_active[self.weld_id]):
            self.weld_ever_active = True
        if self.root_body >= 0:
            # GraspAttachment.follow writes model.body_pos[root_body]; capture
            # the compiled value so a write is detectable.
            if not np.allclose(self.model.body_pos[self.root_body], self._root0):
                self.root_ever_driven = True

    def begin(self):
        """Freeze the invariants the rollout must not change."""
        if self.root_body >= 0:
            self._root0 = self.model.body_pos[self.root_body].copy()

    def run_grip(self, grasped_body=None, steps_per_segment=20):
        trajectory, closing = object_grip_lift_trajectory(
            self.model, self.data, steps_per_segment=steps_per_segment
        )
        self.place_objects_on_ground()
        self.begin()
        self.rollout(
            {"interactor0": trajectory},
            object_bodies=["object"],
            capture_every=10_000,
            substeps=4,
            gripper_opening={"interactor0": closing},
            grasped_body=grasped_body,
            max_controller_speed=0.25,
            controller_free_speed=0.25,
            controller_slowdown_epsilon=0.0,
        )
        return self


class PhysicalGripIsFrictionOnlyTests(unittest.TestCase):
    def test_scene_ships_the_weld_but_leaves_it_inactive(self):
        model = load_model(1, interactor_dynamic=True)
        weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, WELD)
        self.assertGreaterEqual(weld, 0, "the weld should exist for constraint mode")
        self.assertEqual(int(model.eq_type[weld]), int(mujoco.mjtEq.mjEQ_WELD))
        data = mujoco.MjData(model)
        self.assertFalse(
            bool(data.eq_active[weld]),
            "the grasp weld must not be active before a grasp is engaged",
        )

    def test_rigid_physical_grip_never_engages_the_weld(self):
        sim = AttachAudit(
            load_model(1, "object_box.xml", interactor_dynamic=True),
            interactor_dynamic=True,
            width=32,
            height=32,
        )
        sim.run_grip(grasped_body=None)
        self.assertFalse(
            sim.weld_ever_active,
            "physical mode engaged the grasp weld: the lift was not friction-only",
        )

    def test_soft_physical_grip_never_drives_the_flex_frame(self):
        sim = AttachAudit(
            load_model(1, None, BALL_MESH, SOFT, interactor_dynamic=True),
            interactor_dynamic=True,
            width=32,
            height=32,
        )
        self.assertGreater(sim.model.nflexvert, 0, "expected a soft body")
        sim.run_grip(grasped_body=None)
        self.assertFalse(sim.weld_ever_active)
        self.assertFalse(
            sim.root_ever_driven,
            "physical mode drove the flex root frame: the soft lift was not "
            "friction-only",
        )

    def test_constraint_mode_still_engages_the_attach(self):
        # Guards the two tests above against passing vacuously: the same weld
        # must activate when the caller does ask for the attach.
        sim = AttachAudit(
            load_model(1, "object_box.xml", interactor_dynamic=True),
            interactor_dynamic=True,
            width=32,
            height=32,
        )
        sim.run_grip(grasped_body="object")
        self.assertTrue(
            sim.weld_ever_active,
            "constraint mode failed to engage the weld; the physical-mode "
            "checks above would then prove nothing",
        )


if __name__ == "__main__":
    unittest.main()
