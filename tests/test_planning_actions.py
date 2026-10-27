"""Candidate actions rotate rigidly and must reach the object."""

import unittest

import numpy as np

from asid.fim.case import Case
from asid.planning.actions import (
    Action,
    contact_frames,
    generate_actions,
    minimum_clearance,
    require_contact,
    rotate_action,
)


def yaw(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    return rotate_action(points, angle, 0.0, center)


def synthetic_case(frames: int = 4) -> Case:
    axis = np.linspace(-0.1, 0.1, 11)
    x, y = np.meshgrid(axis, axis)
    nodes = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    ring = np.stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, 6, endpoint=False)) * 0.1,
            np.sin(np.linspace(0, 2 * np.pi, 6, endpoint=False)) * 0.1,
            np.full(6, 0.4),
        ],
        axis=1,
    )
    controller = np.repeat(ring[None], frames, axis=0)
    controller[:, :, 2] -= 0.15 * np.arange(frames)[:, None]
    contact = controller[:, :3].copy()
    return Case(
        name="synthetic",
        nodes=nodes.astype(np.float32),
        springs=np.zeros((0, 2), dtype=np.int32),
        rest_lengths=np.zeros(0, dtype=np.float32),
        num_object_springs=0,
        controller=controller.astype(np.float32),
        contact=contact.astype(np.float32),
    )


class RotationTests(unittest.TestCase):
    def test_zero_azimuth_is_the_original(self):
        case = synthetic_case()
        rotated = yaw(case.contact, 0.0, case.nodes.mean(axis=0))
        np.testing.assert_array_equal(rotated, case.contact)

    def test_rotation_preserves_distances_heights_and_center(self):
        case = synthetic_case()
        center = case.nodes.mean(axis=0)
        rotated = yaw(case.contact, np.pi / 3, center).astype(np.float64)
        before = case.contact.astype(np.float64)

        np.testing.assert_allclose(rotated[:, :, 2], before[:, :, 2], atol=1e-7)
        np.testing.assert_allclose(
            np.linalg.norm(rotated - rotated[:, :1], axis=2),
            np.linalg.norm(before - before[:, :1], axis=2),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(rotated[..., :2] - center[:2], axis=2),
            np.linalg.norm(before[..., :2] - center[:2], axis=2),
            atol=1e-6,
        )

    def test_quarter_turn_maps_x_to_y(self):
        points = np.array([[[0.2, 0.0, 0.0]]], dtype=np.float32)
        rotated = yaw(points, np.pi / 2, np.zeros(3))
        np.testing.assert_allclose(rotated, [[[0.0, 0.2, 0.0]]], atol=1e-7)

    def test_yaw_is_a_rotation_about_the_vertical_axis(self):
        case = synthetic_case()
        center = case.nodes.mean(axis=0)
        rotated = rotate_action(case.contact, 0.7, 0.0, center)
        np.testing.assert_allclose(rotated[:, :, 2], case.contact[:, :, 2], atol=1e-7)

    def test_pitch_changes_the_height_and_stays_rigid(self):
        case = synthetic_case()
        center = case.nodes.mean(axis=0)
        before = case.contact.astype(np.float64)
        tilted = rotate_action(case.contact, 0.0, np.deg2rad(-30.0), center).astype(np.float64)

        # A pitch trades the trajectory's vertical offset for a lateral one.
        self.assertGreater(np.ptp(tilted[0, :, 2]), np.ptp(before[0, :, 2]) + 0.05)
        self.assertGreater(abs(tilted.mean(axis=(0, 1))[0]), 0.05)
        np.testing.assert_allclose(
            np.linalg.norm(tilted - tilted[:, :1], axis=2),
            np.linalg.norm(before - before[:, :1], axis=2),
            atol=1e-5,
        )

    def test_yaw_and_pitch_compose_rigidly(self):
        case = synthetic_case()
        center = case.nodes.mean(axis=0)
        composed = rotate_action(case.contact, np.deg2rad(50.0), np.deg2rad(-20.0), center)
        pitched = rotate_action(case.contact, 0.0, np.deg2rad(-20.0), center)
        np.testing.assert_allclose(
            composed, yaw(pitched, np.deg2rad(50.0), center), atol=1e-5
        )


class ContactTests(unittest.TestCase):
    def test_clearance_and_contact_frames(self):
        case = synthetic_case()
        contact = np.array([[[0.0, 0.0, 0.4]], [[0.0, 0.0, 0.1]]], dtype=np.float32)
        np.testing.assert_allclose(minimum_clearance(case.nodes, contact), [0.4, 0.1], atol=1e-6)
        self.assertEqual(contact_frames(case.nodes, contact, 0.2), 1)

    def test_generated_actions_keep_contact(self):
        case = synthetic_case()
        actions = generate_actions(case, np.arange(0, 360, 90))
        self.assertEqual([a.name for a in actions], ["az000", "az090", "az180", "az270"])
        clearance = require_contact(case, actions, radius=0.2)
        self.assertEqual(set(clearance), {a.name for a in actions})

    def test_tilted_grid_names_every_pair_and_keeps_contact(self):
        case = synthetic_case()
        actions = generate_actions(case, [0.0, 90.0], tilts=[0.0, -20.0])
        self.assertEqual(
            [action.name for action in actions],
            ["az000", "az000_t-20", "az090", "az090_t-20"],
        )
        self.assertEqual([round(np.rad2deg(a.tilt)) for a in actions], [0, -20, 0, -20])
        clearance = require_contact(case, actions, radius=0.25)
        self.assertEqual(set(clearance), {action.name for action in actions})
        heights = [action.contact.mean(axis=(0, 1))[2] for action in actions]
        self.assertNotAlmostEqual(heights[0], heights[1], places=3)
        self.assertAlmostEqual(heights[0], heights[2], places=6)

    def test_require_contact_rejects_a_miss(self):
        case = synthetic_case()
        action = Action(
            name="miss",
            azimuth=0.0,
            controller=case.controller + np.float32([2.0, 0.0, 0.0]),
            contact=case.contact + np.float32([2.0, 0.0, 0.0]),
        )
        with self.assertRaises(ValueError):
            require_contact(case, [action], radius=0.2)


if __name__ == "__main__":
    unittest.main()
