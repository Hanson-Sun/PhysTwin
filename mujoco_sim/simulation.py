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

    def set_gripper_opening(self, name: str, closing: float) -> None:
        """Set one claw's normalized hinge closing target in [0, 1]."""
        if not 0.0 <= closing <= 1.0:
            raise ValueError("gripper closing must be between 0 and 1")
        # Increased open_angle from 0.35 to 0.70 rad to provide ~4.5cm clearance per side
        open_angle = 0.70
        close_angle = 0.45
        left_motor = self.model.actuator(f"{name}_finger_l_motor").id
        right_motor = self.model.actuator(f"{name}_finger_r_motor").id
        # The left hinge opens toward +q and closes toward -q; the right
        # hinge opens toward -q and closes toward +q.
        self.data.ctrl[left_motor] = open_angle - (open_angle + close_angle) * closing
        self.data.ctrl[right_motor] = -open_angle + (open_angle + close_angle) * closing

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
        progress=None,
        gripper_opening: dict[str, list[float]] | None = None,
        grasped_body: str | None = None,
        grasp_start_fraction: float = 2.0 / 3.0,
        grasp_offset=(0.0, 0.0, 0.0),
        max_controller_speed: float | None = None,
    ) -> list[Frame]:
        """Play a scripted trajectory and record synthetic RGB-D + ground truth.

        ``max_controller_speed`` limits mocap/interactor translation in metres
        per second. The limit is applied over the full physics interval between
        trajectory commands (``model.opt.timestep * substeps``), so it limits
        the controller rather than changing the object's target trajectory.
        """
        if capture_every < 1:
            raise ValueError("capture_every must be at least 1")
        if substeps < 1:
            raise ValueError("substeps must be at least 1")
        if max_controller_speed is not None and max_controller_speed <= 0.0:
            raise ValueError("max_controller_speed must be positive")
        if not interactor_trajectory:
            raise ValueError("at least one interactor trajectory is required")
        if not 0.0 <= grasp_start_fraction <= 1.0:
            raise ValueError("grasp_start_fraction must be between 0 and 1")
        if grasped_body is not None:
            joint_id = self.model.body(grasped_body).jntadr[0]
            if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"grasped body '{grasped_body}' must have a freejoint")
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]
        else:
            qpos_adr = qvel_adr = None

        names = list(interactor_trajectory.keys())
        lengths = {name: len(trajectory) for name, trajectory in interactor_trajectory.items()}
        if gripper_opening is not None:
            if set(gripper_opening) != set(names):
                raise ValueError("gripper_opening must contain one sequence per interactor")
            if any(len(gripper_opening[name]) != next(iter(lengths.values())) for name in names):
                raise ValueError("gripper opening sequences must match trajectory length")
        if not lengths or min(lengths.values()) == 0:
            raise ValueError("interactor trajectories cannot be empty")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all trajectories must have the same length, got {lengths}")
        n_waypoints = next(iter(lengths.values()))
        frames = []

        for t in range(n_waypoints):
            for name in names:
                target_pos, quat = interactor_trajectory[name][t]
                pos = np.asarray(target_pos, dtype=float)
                if max_controller_speed is not None:
                    mocap_id = self.model.body(name).mocapid[0]
                    if mocap_id < 0:
                        raise ValueError(f"'{name}' is not a mocap body")
                    current_pos = self.data.mocap_pos[mocap_id]
                    max_distance = max_controller_speed * self.model.opt.timestep * substeps
                    delta = pos - current_pos
                    distance = np.linalg.norm(delta)
                    if distance > max_distance:
                        pos = current_pos + delta * (max_distance / distance)
                self.set_interactor_pose(name, pos, quat)
                if gripper_opening is not None:
                    self.set_gripper_opening(name, gripper_opening[name][t])

            if grasped_body is not None and t >= int((n_waypoints - 1) * grasp_start_fraction):
                grasp_position = np.mean(
                    [self.data.mocap_pos[self.model.body(name).mocapid[0]] for name in names],
                    axis=0,
                ) + np.asarray(grasp_offset, dtype=float)
                self.data.qpos[qpos_adr : qpos_adr + 3] = grasp_position
                self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
                mujoco.mj_forward(self.model, self.data)

            self.step(substeps)

            if progress is not None:
                progress(1)

            if t % capture_every == 0:
                frames.append(
                    Frame(
                        time=self.data.time,
                        rgbd=self.render_all_cameras(),
                        object_qpos={b: self.get_free_body_state(b) for b in object_bodies},
                    )
                )

        return frames