"""Export a DigitalTwinSim rollout into PhysTwin's raw case-directory format.

PhysTwin's own data pipeline (script_process_data.py) expects, per case:
  color/<cam_idx>/<frame_idx>.png   RGB frames
  color/<cam_idx>.mp4               RGB video used by dense tracking
  depth/<cam_idx>/<frame_idx>.npy   uint16 depth in millimeters
  calibrate.pkl                     list of 4x4 camera-to-world matrices (OpenCV convention)
  metadata.json                     intrinsics + capture info
  split.json                         train/test frame ranges

The current PhysTwin preprocessing scripts assume exactly three cameras and
use half-open frame ranges in ``split.json``. The exporter enforces that
layout by default.

Everything else (tracking, shape priors, projections) PhysTwin derives itself
from these four things, so this is the full export surface we need to match.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# MuJoCo camera frame: +y up, looks down -z.
# OpenCV camera frame (what PhysTwin/RealSense use): +y down, looks down +z.
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])

# PhysTwin's simulator convention has the ground at z=0 and object points at
# z <= 0.  MuJoCo uses the physically equivalent z-up convention (objects at
# z >= 0), so exported world poses must include this global reflection.  The
# depth images themselves are not vertically flipped; this only converts the
# coordinate frame used by the point-cloud reconstruction and warp solver.
_MJ_WORLD_TO_PHYSTWIN = np.diag([1.0, 1.0, -1.0, 1.0])


def _intrinsics(model, cam_name: str, width: int, height: int) -> list:
    cam_id = model.camera(cam_name).id
    fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
    f = 0.5 * height / np.tan(0.5 * fovy_rad)
    cx, cy = width / 2.0, height / 2.0
    return [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]


def _extrinsic_c2w(model, data, cam_name: str) -> np.ndarray:
    cam_id = model.camera(cam_name).id
    r_mj = data.cam_xmat[cam_id].reshape(3, 3)
    c2w_mj = np.eye(4)
    c2w_mj[:3, :3] = r_mj @ _MJ_TO_CV
    c2w_mj[:3, 3] = data.cam_xpos[cam_id]
    return _MJ_WORLD_TO_PHYSTWIN @ c2w_mj


def export_case(
    sim,
    frames: list,
    case_dir: str,
    fps: int = 30,
    expected_camera_count: int | None = 3,
) -> None:
    """Write `frames` (as returned by DigitalTwinSim.rollout) to `case_dir`
    in PhysTwin's raw format. Cameras are assumed static across the rollout.
    """
    if len(frames) < 2:
        raise ValueError("At least two frames are required for a train/test split")
    if fps <= 0:
        raise ValueError("fps must be positive")

    case_dir = Path(case_dir)
    cams = sim.camera_names
    if not cams:
        raise ValueError("Cannot export a case without cameras")
    if expected_camera_count is not None and len(cams) != expected_camera_count:
        raise ValueError(
            f"PhysTwin export expects {expected_camera_count} cameras, got {len(cams)}"
        )

    writers = []
    for idx in range(len(cams)):
        color_dir = case_dir / "color" / str(idx)
        depth_dir = case_dir / "depth" / str(idx)
        color_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        # A regeneration into an existing case directory is often shorter than
        # the capture it replaces, and stale high-numbered frames would survive
        # to be counted as frames of this capture by the downstream stages.
        for stale in color_dir.glob("*.png"):
            stale.unlink()
        for stale in depth_dir.glob("*.npy"):
            stale.unlink()
        writer = cv2.VideoWriter(
            str(case_dir / "color" / f"{idx}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (sim.width, sim.height),
        )
        if not writer.isOpened():
            for opened_writer in writers:
                opened_writer.release()
            raise RuntimeError(f"Unable to create RGB video for camera {idx}")
        writers.append(writer)

    try:
        for t, frame in enumerate(frames):
            for idx, cam in enumerate(cams):
                rgb, depth = frame.rgbd[cam]
                expected_shape = (sim.height, sim.width)
                if rgb.shape[:2] != expected_shape or rgb.shape[-1] != 3:
                    raise ValueError(
                        f"RGB frame for {cam} has shape {rgb.shape}; expected "
                        f"({sim.height}, {sim.width}, 3)"
                    )
                if depth.shape != expected_shape:
                    raise ValueError(
                        f"Depth frame for {cam} has shape {depth.shape}; expected {expected_shape}"
                    )
                if not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError(f"Depth frame for {cam} contains invalid values")
                Image.fromarray(rgb).save(case_dir / "color" / str(idx) / f"{t}.png")
                writers[idx].write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
                np.save(case_dir / "depth" / str(idx) / f"{t}.npy", depth_mm)
    finally:
        for writer in writers:
            writer.release()

    c2ws = [_extrinsic_c2w(sim.model, sim.data, cam) for cam in cams]
    with open(case_dir / "calibrate.pkl", "wb") as f:
        pickle.dump(c2ws, f)

    frame_num = len(frames)
    metadata = {
        "intrinsics": [_intrinsics(sim.model, cam, sim.width, sim.height) for cam in cams],
        "fps": fps,
        # PhysTwin uses frame_num; num_frames keeps this compatible with the
        # newer RGB-D preparation wrapper as well.
        "frame_num": frame_num,
        "num_frames": frame_num,
        "WH": [sim.width, sim.height],
        "camera_count": len(cams),
        "serial_numbers": [f"sim_cam_{i}" for i in range(len(cams))],
        "depth_unit": "millimeters",
        # How the object was held, so a dataset records whether its lift came
        # from a grasp constraint or from pad friction, and whether the claw was
        # kinematic (mocap) or servo-driven. A mocap claw has infinite mass, so
        # a friction-only grip cannot hold anything on it -- see
        # ``scene._make_claw_dynamic``.
        "sim": {
            "grasp_mode": getattr(sim, "grasp_mode", None),
            "interactor_dynamic": bool(getattr(sim, "interactor_dynamic", False)),
            "grip_force": getattr(sim, "grip_force", None),
            "gripper_closed_angle": getattr(sim, "gripper_closed_angle", None),
        },
    }
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    train_end = int(frame_num * 0.7)
    split = {
        "frame_len": frame_num,
        "train": [0, train_end],
        "test": [train_end, frame_num],
    }
    with open(case_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
