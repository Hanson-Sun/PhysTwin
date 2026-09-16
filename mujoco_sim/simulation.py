"""Thin wrapper around a MuJoCo model for scripted interaction + RGB-D capture."""

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class Frame:
    """One timestep of synthetic data."""

    time: float
    rgbd: dict  # camera_name -> (rgb [H,W,3] uint8, depth [H,W] float32 meters)
    object_qpos: dict  # body_name -> 7-vec (pos + quat), ground truth state


@dataclass
class DigitalTwinSim:
    model: mujoco.MjModel
    width: int = 128
    height: int = 128

    def __post_init__(self):
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.camera_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]
        mujoco.mj_forward(self.model, self.data)

    # ---- control -----------------------------------------------------

    def set_interactor_pose(self, name: str, pos, quat=(1, 0, 0, 0)) -> None:
        """Directly set a mocap-driven interactor's pose (position + wxyz quat)."""
        mocap_id = self.model.body(name).mocapid[0]
        if mocap_id < 0:
            raise ValueError(f"'{name}' is not a mocap body")
        self.data.mocap_pos[mocap_id] = pos
        self.data.mocap_quat[mocap_id] = quat

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # ---- sensing -------------------------------------------------------

    def render_rgbd(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb uint8 [H,W,3], depth float32 [H,W] meters) for one camera."""
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        return rgb, depth

    def render_all_cameras(self) -> dict:
        return {cam: self.render_rgbd(cam) for cam in self.camera_names}

    def get_free_body_state(self, body_name: str) -> np.ndarray:
        """7-vec [pos(3), quat(4)] for a body with a freejoint, e.g. 'object'."""
        adr = self.model.body(body_name).jntadr[0]
        qpos_adr = self.model.jnt_qposadr[adr]
        return self.data.qpos[qpos_adr : qpos_adr + 7].copy()

    # ---- data generation ------------------------------------------------

    def rollout(
        self,
        interactor_trajectory: dict,
        object_bodies: list[str] = ("object",),
        capture_every: int = 1,
        substeps: int = 1,
    ) -> list[Frame]:
        """Play a scripted trajectory and record synthetic RGB-D + ground truth.

        interactor_trajectory: {interactor_name: [(pos, quat), ...]}, all
            lists must share the same length T (one waypoint per outer step).
        Returns one Frame per captured step.
        """
        names = list(interactor_trajectory.keys())
        n_waypoints = len(next(iter(interactor_trajectory.values())))
        frames = []

        for t in range(n_waypoints):
            for name in names:
                pos, quat = interactor_trajectory[name][t]
                self.set_interactor_pose(name, pos, quat)

            self.step(substeps)

            if t % capture_every == 0:
                frames.append(
                    Frame(
                        time=self.data.time,
                        rgbd=self.render_all_cameras(),
                        object_qpos={b: self.get_free_body_state(b) for b in object_bodies},
                    )
                )

        return frames
