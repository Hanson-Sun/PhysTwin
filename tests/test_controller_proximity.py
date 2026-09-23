import unittest

import mujoco
import numpy as np

from mujoco_sim.interactor import (
    controller_local_bounds,
    object_bounds,
    object_grip_lift_trajectory,
    object_push_trajectory,
)
from mujoco_sim.scene import load_model
from mujoco_sim.simulation import DigitalTwinSim


PUSH_CASES = ("object_box.xml", "object_box_heavy_end.xml")


def descendant_geom_ids(sim, body_name: str) -> list[int]:
    body_id = sim.model.body(body_name).id
    geom_ids = []
    for geom_id in range(sim.model.ngeom):
        ancestor = sim.model.geom_bodyid[geom_id]
        while ancestor > 0 and ancestor != body_id:
            ancestor = sim.model.body_parentid[ancestor]
        if ancestor == body_id:
            geom_ids.append(geom_id)
    return geom_ids


def true_clearance(sim, controller: str, body_name: str) -> float:
    """Return the smallest distance between any two of the two bodies' geoms."""
    floor = sim.model.geom("floor").id
    controller_geoms = descendant_geom_ids(sim, controller)
    object_geoms = [g for g in descendant_geom_ids(sim, body_name) if g != floor]
    return min(
        mujoco.mj_geomDistance(sim.model, sim.data, a, b, 1.0, np.zeros(6))
        for a in controller_geoms
        for b in object_geoms
    )


class ControllerProximityTests(unittest.TestCase):
    def test_rope_bounds_include_flexible_vertices(self):
        sim = DigitalTwinSim(load_model(1, "object_rope.xml"))

        lower, upper = sim._object_bounds("object")

        extents = upper - lower
        if not float(extents.max()) > 0.2:
            bodies = [sim.model.body(i).name for i in range(sim.model.nbody)]
            geoms = [sim.model.geom(i).name for i in range(sim.model.ngeom)]
            self.fail(
                f"bounds={lower}, {upper}, nflexvert={sim.model.nflexvert}, "
                f"bodies={bodies}, geoms={geoms}"
            )
        self.assertGreater(float(np.count_nonzero(extents > 0.01)), 1)

    def test_heavy_end_push_has_free_horizontal_approach(self):
        sim = DigitalTwinSim(load_model(1, "object_box_heavy_end.xml"))
        trajectory = object_push_trajectory(
            sim.model, sim.data, steps_per_segment=2, margin=0.06
        )

        horizontal = next(
            (
                (np.asarray(start[0]), np.asarray(end[0]))
                for start, end in zip(trajectory[:-1], trajectory[1:])
                if abs(start[0][2] - end[0][2]) < 1e-12
                and abs(start[0][0] - end[0][0]) > 1e-12
            ),
            None,
        )
        self.assertIsNotNone(horizontal)
        start, end = horizontal
        object_center, object_rotation, object_half = sim._object_oriented_bounds(
            "object"
        )
        interval = sim._segment_proximity_interval(
            start,
            end,
            object_center,
            object_rotation,
            object_half,
            sim._controller_geom_boxes("interactor0"),
            0.0,
        )

        # The sweep starts clear of the object, enters contact, and leaves it
        # again before the retract waypoint.
        self.assertIsNotNone(interval)
        self.assertGreater(interval[0], 0.0)
        self.assertLess(interval[1], 1.0)

    def test_push_margin_is_geometry_clearance(self):
        model = load_model(1, "object_box.xml")
        sim = DigitalTwinSim(model)
        lower, upper = object_bounds(model, sim.data)
        controller_lower, controller_upper = controller_local_bounds(
            model, sim.data
        )
        margin = 0.06
        trajectory = object_push_trajectory(
            model, sim.data, steps_per_segment=2, margin=margin
        )

        start = np.asarray(trajectory[0][0])
        end = np.asarray(trajectory[-1][0])
        self.assertLessEqual(
            start[0] + controller_upper[0], lower[0] - margin + 1e-9
        )
        self.assertGreaterEqual(
            end[0] + controller_lower[0], upper[0] + margin - 1e-9
        )

    def test_controller_envelope_is_used_instead_of_mocap_point(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"))
        controller_lower, controller_upper = sim._controller_local_bounds("interactor0")
        object_lower, object_upper = sim._object_bounds("object")

        # A translated claw can overlap the box even when its mocap origin is
        # outside the object AABB. The interval is derived from the actual
        # controller envelope, not an arbitrary object-radius inflation.
        proximity_lower = object_lower - controller_upper
        proximity_upper = object_upper - controller_lower
        center = np.array([0.0, 0.0, 0.11])
        self.assertTrue(np.all(center >= proximity_lower))
        self.assertTrue(np.all(center <= proximity_upper))
        self.assertFalse(
            np.all(center >= object_lower) and np.all(center <= object_upper)
        )

    def test_dynamic_step_uses_free_speed_until_proximity(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"))
        current = {"interactor0": np.array([-0.4, 0.0, 1.0])}
        targets = {"interactor0": np.array([0.4, 0.0, 1.0])}

        distance = sim._step_distance(
            current, targets, ["interactor0"], 0.1, 1.0, 1, 0.0, "object"
        )

        # One millisecond of sim time at the free speed.
        self.assertAlmostEqual(distance, 0.001, places=9)

    def test_dynamic_bounds_follow_object_motion(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"))
        current = {"interactor0": np.array([-0.4, 0.0, 0.10])}
        targets = {"interactor0": np.array([0.4, 0.0, 0.10])}
        names = ["interactor0"]

        before = sim._step_distance(
            current, targets, names, 0.1, 1.0, 1, 0.0, "object"
        )
        joint_id = sim.model.body("object").jntadr[0]
        qpos_adr = sim.model.jnt_qposadr[joint_id]
        sim.data.qpos[qpos_adr] = -0.4
        mujoco.mj_forward(sim.model, sim.data)
        after = sim._step_distance(
            current, targets, names, 0.1, 1.0, 1, 0.0, "object"
        )

        self.assertLess(after, before)

    def test_controller_envelope_uses_true_geom_extents(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"))
        lower, upper = sim._controller_local_bounds("interactor0")

        # The claw's fingers are ~1.8 cm thick capsules; using the bounding
        # sphere radius per axis used to report 7.9 cm on the thin axes, which
        # widened the slowdown zone by ~4 cm on every side.
        self.assertAlmostEqual(float(upper[0]), 0.085, places=3)
        self.assertAlmostEqual(float(upper[1]), 0.035, places=3)
        self.assertAlmostEqual(float(lower[1]), -0.035, places=3)

    def test_controller_envelope_is_per_geom_box_union(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"))
        boxes = sim._controller_geom_boxes("interactor0")

        self.assertEqual(len(boxes), len(descendant_geom_ids(sim, "interactor0")))
        lower, upper = sim._controller_local_bounds("interactor0")
        box_lower = np.min(
            [center - np.abs(rotation) @ half for center, rotation, half in boxes],
            axis=0,
        )
        box_upper = np.max(
            [center + np.abs(rotation) @ half for center, rotation, half in boxes],
            axis=0,
        )
        np.testing.assert_allclose(box_lower, lower, atol=1e-9)
        np.testing.assert_allclose(box_upper, upper, atol=1e-9)
        # Each claw geom is a rotated box centred on its own geometry.
        for _, rotation, half in boxes:
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
            self.assertTrue(np.all(half > 0.0))

    def test_object_oriented_box_matches_rotated_plank(self):
        sim = DigitalTwinSim(load_model(1, "object_box_heavy_end.xml"))
        _, rotation, half = sim._object_oriented_bounds("object")
        lower, upper = sim._object_bounds("object")

        # The plank is 0.36 x 0.08 x 0.05 m in its own frame.
        np.testing.assert_allclose(
            np.sort(half), [0.025, 0.04, 0.18], atol=3e-3
        )
        # Its axis-aligned hull is only that wide on x because it is rotated.
        self.assertGreater(float(upper[0] - lower[0]), 0.18)
        # The box is attached to the body, so it turns with the object.
        self.assertGreater(abs(float(rotation[0, 1])), 0.5)

    def test_non_rigid_object_keeps_axis_aligned_bounds(self):
        sim = DigitalTwinSim(load_model(1, "object_rope.xml"))
        center, rotation, half = sim._object_oriented_bounds("object")
        lower, upper = sim._object_bounds("object")

        # A cable's links are not descendants of the compatibility object body.
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(center, (lower + upper) * 0.5)
        np.testing.assert_allclose(half, (upper - lower) * 0.5)

    def proximity_trigger(self, case: str, epsilon: float = 0.02):
        """Return (claw x, true clearance) where the proximity test first fires."""
        steps_per_segment = 300
        sim = DigitalTwinSim(load_model(1, case), width=16, height=16)
        trajectory = object_push_trajectory(
            sim.model, sim.data, steps_per_segment=steps_per_segment
        )
        sim.place_objects_on_ground()
        contact_z = float(trajectory[steps_per_segment][0][2])
        mocap_id = sim.model.body("interactor0").mocapid[0]

        for x in np.arange(-0.30, 0.001, 0.005):
            sim.data.mocap_pos[mocap_id] = [x, 0.0, contact_z]
            sim.data.mocap_quat[mocap_id] = [1, 0, 0, 0]
            mujoco.mj_forward(sim.model, sim.data)
            center, rotation, half = sim._object_oriented_bounds("object")
            point = np.array([x, 0.0, contact_z])
            fired = sim._segment_proximity_interval(
                point,
                point,
                center,
                rotation,
                half,
                sim._controller_geom_boxes("interactor0"),
                epsilon,
            )
            if fired is not None:
                return x, true_clearance(sim, "interactor0", "object")
        self.fail(f"proximity test never triggered for {case}")

    def test_slowdown_starts_only_near_real_contact(self):
        for case in PUSH_CASES:
            with self.subTest(case=case):
                _, clearance = self.proximity_trigger(case)
                self.assertGreater(clearance, -0.01)
                self.assertLess(clearance, 0.04)

    def test_slowdown_zone_is_consistent_across_object_shapes(self):
        flat_case, flat_clearance = self.proximity_trigger("object_box.xml")
        heavy_case, heavy_clearance = self.proximity_trigger(
            "object_box_heavy_end.xml"
        )

        # The rotated heavy-end plank is 0.099 m wide on x per side only at
        # |y| = 0.16 m. Bounding it with that world hull used to hold the claw
        # at max_controller_speed from ~0.15 m away, while the axis-aligned box
        # released it at ~0.09 m.
        self.assertLess(abs(flat_clearance - heavy_clearance), 0.02)
        self.assertGreater(heavy_case, flat_case - 0.1)

    def record_commands(self, sim):
        """Wrap ``step`` so the test can see each command's commanded pose."""
        poses = []
        original_step = sim.step

        def step(n=1):
            poses.append(sim.data.mocap_pos[0].copy())
            original_step(n)

        sim.step = step
        return poses

    def test_rollout_spends_its_whole_step_allowance(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=16)
        spacing = 0.001
        waypoints = [(0.0, 0.0, 0.6 + index * spacing) for index in range(200)]
        trajectory = [(point, (1.0, 0.0, 0.0, 0.0)) for point in waypoints]
        poses = self.record_commands(sim)

        sim.rollout(
            {"interactor0": trajectory},
            capture_every=10**6,
            substeps=4,
            max_controller_speed=0.25,
            controller_free_speed=0.7,
        )

        travel = np.linalg.norm(np.diff(np.asarray(poses), axis=0), axis=1)
        free_distance = 0.7 * sim.model.opt.timestep * 4
        # Waypoints this close used to pin the claw to one per command.
        self.assertGreater(float(np.median(travel)), 0.9 * free_distance)
        self.assertLessEqual(float(travel.max()), free_distance + 1e-9)
        np.testing.assert_allclose(poses[-1], waypoints[-1], atol=1e-9)

    def test_repeated_waypoints_still_cost_one_command_each(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=16)
        hold = 40
        start = tuple(sim.data.mocap_pos[0])
        trajectory = [(start, (1.0, 0.0, 0.0, 0.0))] * hold
        poses = self.record_commands(sim)

        sim.rollout(
            {"interactor0": trajectory},
            capture_every=10**6,
            substeps=4,
            max_controller_speed=0.25,
            controller_free_speed=0.7,
        )

        # Holds are expressed as repeated waypoints, so they must keep their
        # duration instead of collapsing into a single command.
        self.assertEqual(len(poses), hold)
        np.testing.assert_allclose(
            np.asarray(poses), np.tile(start, (hold, 1)), atol=1e-12
        )

    def grip_lift_rollout(self, sim, steps_per_segment=60, grasp_only=False):
        """Play a grip_lift grasp and return per-command claw/object state."""
        trajectory, closing = object_grip_lift_trajectory(
            sim.model, sim.data, steps_per_segment=steps_per_segment
        )
        sim.place_objects_on_ground()
        motors = [
            sim.model.actuator(f"interactor0_finger_{side}_motor").id
            for side in "lr"
        ]
        hinge = sim.model.joint("interactor0_finger_l_hinge").id
        pads = [sim.model.geom(f"interactor0_pad_{side}").id for side in "lr"]
        state = []
        original_step = sim.step

        def step(n=1):
            state.append(
                [float(sim.data.ctrl[model_id]) for model_id in motors]
                + [
                    float(sim.data.qpos[sim.model.jnt_qposadr[hinge]]),
                    float(sim.data.geom_xpos[pads[0]][0]),
                    float(sim.data.geom_xpos[pads[1]][0]),
                ]
            )
            original_step(n)

        sim.step = step
        # The grasp is force-attached from the lift onwards, so the grasp-only
        # run stops at the end of the hold to observe the box being gripped.
        end = 2 * steps_per_segment if grasp_only else len(trajectory)
        sim.rollout(
            {"interactor0": trajectory[:end]},
            capture_every=10**6,
            substeps=4,
            gripper_opening={"interactor0": closing[:end]},
            grasped_body=None if grasp_only else "object",
            grasp_offset=(0.0, 0.0, 0.0),
            max_controller_speed=0.25,
            controller_free_speed=0.25,
            controller_slowdown_epsilon=0.12,
        )
        return np.asarray(state), sim

    def test_gripper_closes_gradually_across_the_hold(self):
        steps_per_segment = 60
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=16)
        state, _ = self.grip_lift_rollout(
            sim, steps_per_segment=steps_per_segment, grasp_only=True
        )
        controls = state[:, 0]

        # The hold is a run of repeated waypoints. Its gripper closing used to
        # be applied only on the first command that moved again, so the fingers
        # snapped shut in a single 4 ms command while the lift started.
        self.assertLess(float(np.abs(np.diff(controls)).max()), 0.02)
        closing_commands = np.count_nonzero(
            (controls < controls[0] - 1e-9) & (controls > controls[-1] + 1e-9)
        )
        self.assertGreater(closing_commands, steps_per_segment - 5)
        self.assertLess(controls[-1], controls[0])
        np.testing.assert_allclose(state[:, 0], -state[:, 1], atol=1e-9)

    def test_gripper_pads_stop_at_the_object_faces(self):
        steps_per_segment = 60
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=16)
        state, sim = self.grip_lift_rollout(
            sim, steps_per_segment=steps_per_segment, grasp_only=True
        )

        lower, upper = sim._object_bounds("object")
        object_width = float(upper[0] - lower[0])
        pad_radius = float(sim.model.geom("interactor0_pad_l").size[0])
        hinge, pad_l, pad_r = state[:, 2], state[:, 3], state[:, 4]
        pad_gap = pad_r - pad_l
        grip = slice(-steps_per_segment, None)
        object_state = sim.get_free_body_state("object")

        # A target that closes past the pads' contact angle drives the pads
        # through the object, which squeezes it out of the grip instead of
        # holding it. On the grip the pads stop on the faces: their centres stay
        # a full object width apart, so they never cross it or each other.
        self.assertGreaterEqual(float(pad_gap[grip].min()), object_width)
        self.assertLess(float(pad_gap[grip].min()) - 2.0 * pad_radius, object_width)
        self.assertLess(float(pad_l.max()), 0.0)
        self.assertGreater(float(pad_r.min()), 0.0)
        # The fingers never swing past vertical, and the box is held in place
        # rather than pushed or ejected by the closing pads.
        self.assertGreater(float(hinge[grip].min()), 0.0)
        np.testing.assert_allclose(object_state[:2], [0.0, 0.0], atol=2e-3)
        self.assertAlmostEqual(float(object_state[2]), 0.05, delta=2e-3)

    def test_rotated_object_does_not_trigger_before_reach(self):
        sim = DigitalTwinSim(
            load_model(1, "object_box_heavy_end.xml"), width=16, height=16
        )
        trajectory = object_push_trajectory(
            sim.model, sim.data, steps_per_segment=2, margin=0.06
        )
        sim.place_objects_on_ground()
        contact_z = float(trajectory[2][0][2])
        origin = np.array([-0.18, 0.0, contact_z])

        # The union of both world-axis-aligned hulls still reports contact here,
        # even though the closest claw geometry is 9 cm from the plank.
        lower, upper = sim._object_bounds("object")
        controller_lower, controller_upper = sim._controller_local_bounds(
            "interactor0"
        )
        self.assertIsNotNone(
            sim._segment_proximity_interval(
                origin,
                origin,
                (lower + upper) * 0.5,
                np.eye(3),
                (upper - lower) * 0.5,
                [
                    (
                        (controller_lower + controller_upper) * 0.5,
                        np.eye(3),
                        (controller_upper - controller_lower) * 0.5,
                    )
                ],
                0.02,
            )
        )

        center, rotation, half = sim._object_oriented_bounds("object")
        self.assertIsNone(
            sim._segment_proximity_interval(
                origin,
                origin,
                center,
                rotation,
                half,
                sim._controller_geom_boxes("interactor0"),
                0.02,
            )
        )
        self.assertGreater(true_clearance(sim, "interactor0", "object"), 0.06)


if __name__ == "__main__":
    unittest.main()
