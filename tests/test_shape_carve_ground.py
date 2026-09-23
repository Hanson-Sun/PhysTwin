import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from data_process.shape_carve import flatten_mesh_to_ground


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CASES = (
    "sim_rigid_box",
    "sim_rigid_box_heavy_end",
    "sim_rope",
    # This is the lifted case from the reported traceback.
    "sim_rigid_box_grip_lift",
)


class GroundFlatteningTests(unittest.TestCase):
    def test_mesh_entirely_above_ground_is_not_clipped(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh.apply_translation([0.0, 0.0, 2.0])

        flattened = flatten_mesh_to_ground(mesh, 0.0)

        self.assertIs(flattened, mesh)
        self.assertTrue(mesh.is_watertight)
        np.testing.assert_array_equal(flattened.vertices, mesh.vertices)
        np.testing.assert_array_equal(flattened.faces, mesh.faces)

    def test_mesh_entirely_below_ground_is_not_clipped(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh.apply_translation([0.0, 0.0, -2.0])

        flattened = flatten_mesh_to_ground(mesh, 0.0)

        self.assertIs(flattened, mesh)
        self.assertTrue(mesh.is_watertight)

    def test_straddling_box_is_clipped_and_capped(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        flattened = flatten_mesh_to_ground(mesh, 0.0)

        self.assertIsNot(flattened, mesh)
        self.assertTrue(flattened.is_watertight)
        self.assertLessEqual(float(flattened.vertices[:, 2].max()), 1e-7)
        self.assertAlmostEqual(float(flattened.bounds[1, 2]), 0.0, places=6)

    def test_non_simple_cap_uses_mesh_repair_fallback(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        with patch(
            "data_process.shape_carve._triangulate_boundary_loop",
            side_effect=RuntimeError("non-simple test boundary"),
        ):
            flattened = flatten_mesh_to_ground(mesh, 0.0)

        self.assertTrue(flattened.is_watertight)
        self.assertLessEqual(float(flattened.vertices[:, 2].max()), 1e-7)

    def test_representative_simulation_meshes_do_not_fail(self):
        for case_name in REPRESENTATIVE_CASES:
            path = REPO_ROOT / "data" / "different_types" / case_name / "shape" / "object.glb"
            with self.subTest(case_name=case_name):
                self.assertTrue(path.exists(), path)
                mesh = trimesh.load_mesh(path, process=False)
                self.assertIsInstance(mesh, trimesh.Trimesh)
                before = mesh.vertices.copy()
                flattened = flatten_mesh_to_ground(mesh, 0.0)
                self.assertTrue(np.isfinite(flattened.vertices).all())
                if float(before[:, 2].min()) >= -1e-7:
                    self.assertIs(flattened, mesh)
                    np.testing.assert_array_equal(flattened.vertices, before)


if __name__ == "__main__":
    unittest.main()
