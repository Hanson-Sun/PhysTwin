"""MuJoCo trajectory adaptor tests."""

import unittest

import numpy as np

from asid.planning.mujoco import execute
from asid.planning.trajectory import Trajectory, plan_trajectory, rigid_template
from mujoco_sim.scene import load_model
from mujoco_sim.simulation import DigitalTwinSim


class MuJoCoTrajectoryTests(unittest.TestCase):
    def test_translation_is_reflected_and_timed(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=12)
        initial = sim.commanded_interactor_pose("interactor0")
        quat = sim.data.mocap_quat[sim.model.body("interactor0").mocapid[0]].copy()
        offsets = np.array([[0, 0, 0], [0.01, 0, -0.02], [0.02, 0, -0.03]])
        points = np.array([[0.1, 0.2, -0.1], [0.11, 0.2, -0.1]])
        controller, contact = rigid_template(points, points, offsets)
        dt = 2 * sim.model.opt.timestep
        plan = plan_trajectory(
            controller, contact, lambda t: float(t.offsets[-1, 0]),
            lower=np.full(3, -1.0), upper=np.ones(3), max_speed=100.0,
            dt=dt, offset_limit=0.001, knots=1, candidates=2, seed=2,
        )
        result = execute(sim, plan.trajectory, initial_position=initial, quaternion=quat, frame_dt=dt)
        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.commands.shape, (4, 3))
        expected = initial + (offsets[-1] + plan.trajectory.offsets[-1]) * [1, 1, -1]
        np.testing.assert_allclose(result.commands[-1], expected)
        np.testing.assert_allclose(result.times, np.arange(1, 5) * sim.model.opt.timestep)
        np.testing.assert_allclose(sim.commanded_interactor_pose("interactor0"), result.commands[-1])
        self.assertEqual(len(result.frames[0].rgbd), len(sim.camera_names))

    def test_nonrigid_motion_is_rejected(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=12)
        initial = sim.commanded_interactor_pose("interactor0")
        quat = sim.data.mocap_quat[sim.model.body("interactor0").mocapid[0]].copy()
        points = np.zeros((2, 2, 3))
        points[1, 0, 0] = 0.1
        trajectory = Trajectory(np.zeros((2, 3)), points, points)
        with self.assertRaisesRegex(ValueError, "rigid translation"):
            execute(sim, trajectory, initial_position=initial, quaternion=quat, frame_dt=sim.model.opt.timestep)


if __name__ == "__main__":
    unittest.main()
