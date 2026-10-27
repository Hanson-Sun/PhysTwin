"""Run directories record configs, arrays, and a reloadable manifest."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from asid.fim.logging_utils import Run


class RunDirectoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_writes_config(self):
        run = Run.create(self.root, "case_a", "run_1", {"seed": 0, "g": 100})
        self.assertEqual(run.path, self.root / "case_a" / "run_1")
        config = json.loads((run.path / "config.json").read_text())
        self.assertEqual(config, {"seed": 0, "g": 100})

    def test_arrays_are_recorded_in_the_manifest(self):
        run = Run.create(self.root, "case_a", "run_1", {})
        run.save_array("nominal/nodes.npy", np.zeros((4, 3), dtype=np.float32), "nodes")
        manifest = json.loads(run.close().read_text())

        entry = manifest["arrays"]["nominal/nodes.npy"]
        self.assertEqual(entry["role"], "nodes")
        self.assertEqual(entry["shape"], [4, 3])
        self.assertEqual(entry["dtype"], "float32")
        self.assertEqual(manifest["case"], "case_a")
        self.assertEqual(manifest["run_id"], "run_1")
        self.assertEqual(manifest["config"], "config.json")
        np.testing.assert_array_equal(np.load(run.path / "nominal/nodes.npy"), np.zeros((4, 3)))

    def test_rejects_path_escape(self):
        run = Run.create(self.root, "case_a", "run_1", {})
        with self.assertRaises(ValueError):
            run.save_array("../outside.npy", np.zeros(1), "bad")

    def test_rejects_unsafe_names(self):
        with self.assertRaises(ValueError):
            Run.create(self.root, "case/other", "run_1", {})
        with self.assertRaises(ValueError):
            Run.create(self.root, "case_a", "../run_1", {})


if __name__ == "__main__":
    unittest.main()
