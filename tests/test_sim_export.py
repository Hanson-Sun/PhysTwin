import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mujoco_sim.phystwin_export import export_case
from mujoco_sim.scene import load_model
from mujoco_sim.simulation import DigitalTwinSim, Frame


def fake_capture(sim, frame_num: int) -> list[Frame]:
    rgb = np.zeros((sim.height, sim.width, 3), dtype=np.uint8)
    depth = np.full((sim.height, sim.width), 0.5, dtype=np.float32)
    return [
        Frame(
            time=index * 0.01,
            rgbd={camera: (rgb, depth) for camera in sim.camera_names},
            object_qpos={"object": np.zeros(7)},
        )
        for index in range(frame_num)
    ]


class SimExportTests(unittest.TestCase):
    def test_shorter_reexport_clears_stale_frames(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=12)
        stale_frames = 7

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            export_case(sim, fake_capture(sim, stale_frames), case_dir, fps=30)
            self.assertEqual(len(list((case_dir / "color" / "0").glob("*.png"))), stale_frames)
            self.assertEqual(len(list((case_dir / "depth" / "0").glob("*.npy"))), stale_frames)

            # A shorter regeneration must not leave the old high-numbered frames
            # behind: the downstream stages count them as frames of the new
            # capture and then disagree with the tracked video length.
            frame_num = 3
            export_case(sim, fake_capture(sim, frame_num), case_dir, fps=30)

            for camera in range(len(sim.camera_names)):
                frames = sorted(
                    int(path.stem) for path in (case_dir / "color" / str(camera)).glob("*.png")
                )
                depths = sorted(
                    int(path.stem) for path in (case_dir / "depth" / str(camera)).glob("*.npy")
                )
                self.assertEqual(frames, list(range(frame_num)))
                self.assertEqual(depths, list(range(frame_num)))

            metadata = json.loads((case_dir / "metadata.json").read_text())
            self.assertEqual(metadata["frame_num"], frame_num)
            self.assertEqual(metadata["num_frames"], frame_num)

    def test_export_frame_count_matches_written_frames(self):
        sim = DigitalTwinSim(load_model(1, "object_box.xml"), width=16, height=12)

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            frame_num = 5
            export_case(sim, fake_capture(sim, frame_num), case_dir, fps=30)

            metadata = json.loads((case_dir / "metadata.json").read_text())
            split = json.loads((case_dir / "split.json").read_text())
            self.assertEqual(metadata["frame_num"], split["frame_len"])
            self.assertEqual(split["test"][1], frame_num)


if __name__ == "__main__":
    unittest.main()
