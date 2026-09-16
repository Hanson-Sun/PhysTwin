"""Export a DigitalTwinSim rollout into PhysTwin's raw case-directory format.

PhysTwin's own data pipeline (script_process_data.py) expects, per case:
  color/<cam_idx>/<frame_idx>.png   RGB frames
  depth/<cam_idx>/<frame_idx>.png   uint16 depth in millimeters
  calibrate.pkl                     list of 4x4 camera-to-world matrices (OpenCV convention)
  metadata.json                     intrinsics + capture info

Everything else (tracking, shape priors, projections) PhysTwin derives itself
from these four things, so this is the full export surface we need to match.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

# MuJoCo camera frame: +y up, looks down -z.
# OpenCV camera frame (what PhysTwin/RealSense use): +y down, looks down +z.
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def _intrinsics(model, cam_name: str, width: int, height: int) -> list:
    cam_id = model.camera(cam_name).id
    fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
    f = 0.5 * height / np.tan(0.5 * fovy_rad)
    cx, cy = width / 2.0, height / 2.0
    return [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]


def _extrinsic_c2w(model, data, cam_name: str) -> np.ndarray:
    cam_id = model.camera(cam_name).id
    r_mj = data.cam_xmat[cam_id].reshape(3, 3)
    c2w = np.eye(4)
    c2w[:3, :3] = r_mj @ _MJ_TO_CV
    c2w[:3, 3] = data.cam_xpos[cam_id]
    return c2w


def export_case(sim, frames: list, case_dir: str, fps: int = 30) -> None:
    """Write `frames` (as returned by DigitalTwinSim.rollout) to `case_dir`
    in PhysTwin's raw format. Cameras are assumed static across the rollout.
    """
    case_dir = Path(case_dir)
    cams = sim.camera_names

    for idx in range(len(cams)):
        (case_dir / "color" / str(idx)).mkdir(parents=True, exist_ok=True)
        (case_dir / "depth" / str(idx)).mkdir(parents=True, exist_ok=True)

    for t, frame in enumerate(frames):
        for idx, cam in enumerate(cams):
            rgb, depth = frame.rgbd[cam]
            Image.fromarray(rgb).save(case_dir / "color" / str(idx) / f"{t}.png")
            depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(case_dir / "depth" / str(idx) / f"{t}.png")

    c2ws = [_extrinsic_c2w(sim.model, sim.data, cam) for cam in cams]
    with open(case_dir / "calibrate.pkl", "wb") as f:
        pickle.dump(c2ws, f)

    metadata = {
        "intrinsics": [_intrinsics(sim.model, cam, sim.width, sim.height) for cam in cams],
        "fps": fps,
        "frame_num": len(frames),
        "WH": [sim.width, sim.height],
        "camera_count": len(cams),
        "serial_numbers": [f"sim_cam_{i}" for i in range(len(cams))],
    }
    with open(case_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
