"""Warm-start input and checkpoint compatibility checks."""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.continue_warp import prepare_data


class PrefixTests(unittest.TestCase):
    def test_prefixes_preserve_point_identity_and_geometry(self):
        data = {
            "object_points": np.arange(6 * 5 * 3).reshape(6, 5, 3),
            "object_colors": np.zeros((6, 5, 3)),
            "object_visibilities": np.ones((6, 5), bool),
            "object_motions_valid": np.ones((6, 5), bool),
            "controller_points": np.zeros((6, 2, 3)),
            "controller_points_dense": np.zeros((6, 3, 3)),
            "surface_points": np.ones((4, 3)), "interior_points": np.ones((4, 3)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            with source.open("wb") as file:
                pickle.dump(data, file)
            info1 = prepare_data(source, root / "one.pkl", 3, 3)
            info2 = prepare_data(source, root / "two.pkl", 5, 3)
            self.assertEqual(info1["indices"], info2["indices"])
            with (root / "one.pkl").open("rb") as file:
                first = pickle.load(file)
            with (root / "two.pkl").open("rb") as file:
                second = pickle.load(file)
            np.testing.assert_array_equal(first["object_points"], second["object_points"][:3])
            self.assertEqual(first["interior_points"].shape, (0, 3))
            with self.assertRaises(FileExistsError):
                prepare_data(source, root / "one.pkl", 3, 3)


class ResumeTests(unittest.TestCase):
    def test_changed_geometry_is_rejected_before_parameters_are_loaded(self):
        from qqtt.engine.trainer_warp_upstream import InvPhyTrainerWarp
        from qqtt.utils import cfg

        class Simulator:
            n_springs = 1

        trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
        trainer.init_vertices = torch.zeros((2, 3))
        trainer.init_springs = torch.zeros((1, 2), dtype=torch.int32)
        trainer.init_rest_lengths = torch.ones(1)
        trainer.init_masses = torch.ones(2)
        trainer.controller_points_dense = None
        trainer.simulator = Simulator()
        trainer.trainable_parameters = []
        checkpoint = {
            "resume_geometry": {"vertices": torch.ones((2, 3)), "springs": trainer.init_springs,
                                "rest_lengths": trainer.init_rest_lengths},
            "trainable_names": [], "trainable_values": {}, "optimizer_state_dict": {}, "epoch": 0,
        }
        with patch.object(cfg, "device", "cpu", create=True), tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "vertices"):
                trainer.resume_training(path)

    def test_resume_restores_raw_parameters_and_adam_state(self):
        from qqtt.engine.trainer_warp_upstream import InvPhyTrainerWarp
        from qqtt.utils import cfg

        trainer = InvPhyTrainerWarp.__new__(InvPhyTrainerWarp)
        trainer.init_vertices = torch.zeros((2, 3))
        trainer.init_springs = torch.zeros((1, 2), dtype=torch.int32)
        trainer.init_rest_lengths = torch.ones(1)
        trainer.init_masses = torch.ones(2)
        trainer.controller_points_dense = None
        trainer.num_object_springs = 1
        trainer.simulator = SimpleNamespace(dt=0.01, num_substeps=2, mass_normalization=1.0, learn_mass=False)
        parameter = torch.tensor([-0.5], requires_grad=True)
        trainer.trainable_parameters = [("collide_fric", parameter)]
        trainer.optimizer = torch.optim.Adam([parameter], lr=0.01)
        source = torch.tensor([2.5], requires_grad=True)
        optimizer = torch.optim.Adam([source], lr=0.01)
        source.grad = torch.ones_like(source)
        optimizer.step()
        checkpoint = {
            "resume_geometry": {"vertices": trainer.init_vertices, "springs": trainer.init_springs,
                                "rest_lengths": trainer.init_rest_lengths, "masses": trainer.init_masses,
                                "contact_points": None},
            "resume_settings": {"dt": 0.01, "num_substeps": 2, "mass_normalization": 1.0, "learn_mass": False},
            "num_object_springs": 1, "trainable_names": ["collide_fric"],
            "trainable_values": {"collide_fric": source.detach().clone()},
            "optimizer_state_dict": optimizer.state_dict(), "epoch": 4,
        }
        with patch.object(cfg, "device", "cpu", create=True), tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            torch.save(checkpoint, path)
            self.assertEqual(trainer.resume_training(path), 4)
        torch.testing.assert_close(parameter, source)
        trainer.optimizer.zero_grad()
        optimizer.zero_grad()
        parameter.grad = torch.ones_like(parameter)
        source.grad = torch.ones_like(source)
        trainer.optimizer.step()
        optimizer.step()
        torch.testing.assert_close(parameter, source)


if __name__ == "__main__":
    unittest.main()
