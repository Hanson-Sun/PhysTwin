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

    def _object_bounds(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative world-space bounds for an object's geoms."""
        body_id = self.model.body(body_name).id
        points = []
        for geom_id in range(self.model.ngeom):
            ancestor = self.model.geom_bodyid[geom_id]
            while ancestor > 0 and ancestor != body_id:
                ancestor = self.model.body_parentid[ancestor]
            if ancestor == body_id:
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                    extents = np.abs(
                        self.data.geom_xmat[geom_id].reshape(3, 3)
                    ) @ self.model.geom_size[geom_id]
                else:
                    extents = np.full(3, self.model.geom_rbound[geom_id])
                points.extend(
                    [
                        self.data.geom_xpos[geom_id] - extents,
                        self.data.geom_xpos[geom_id] + extents,
                    ]
                )
        if not points:
            raise ValueError(f"could not find geometry for body '{body_name}'")
        points = np.asarray(points, dtype=float)
        return points.min(axis=0), points.max(axis=0)

    @staticmethod
    def _segment_bounds_interval(
        start: np.ndarray,
        end: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        epsilon: float,
    ) -> tuple[float, float] | None:
        """Return the parameter interval inside an epsilon-expanded AABB."""
        lower, upper = lower - epsilon, upper + epsilon
        direction = end - start
        entry, exit = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return None
                continue
            near = (lower[axis] - start[axis]) / direction[axis]
            far = (upper[axis] - start[axis]) / direction[axis]
            if near > far:
                near, far = far, near
            entry, exit = max(entry, near), min(exit, far)
            if entry > exit:
                return None
        return entry, exit

    def place_objects_on_ground(self, ground_z: float = 0.0) -> None:
        """Place every non-controller object root on the ground before frame 0."""
        root_geoms = {}
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            if self.model.geom(geom_id).name == "floor":
                continue
            ancestor = body_id
            is_controller = False
            while ancestor > 0:
                if self.model.body(ancestor).mocapid[0] >= 0:
                    is_controller = True
                    break
                ancestor = self.model.body_parentid[ancestor]
            if is_controller:
                continue

            root = body_id
            while self.model.body_parentid[root] > 0:
                root = self.model.body_parentid[root]
            root_geoms.setdefault(root, []).append(geom_id)

        for root, geom_ids in root_geoms.items():
            lower = np.inf
            for geom_id in geom_ids:
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                    extents = np.abs(
                        self.data.geom_xmat[geom_id].reshape(3, 3)
                    ) @ self.model.geom_size[geom_id]
                else:
                    extents = np.full(3, self.model.geom_rbound[geom_id])
                lower = min(lower, self.data.geom_xpos[geom_id, 2] - extents[2])
            delta = ground_z - lower
            root_joint = self.model.body(root).jntadr[0]
            if root_joint >= 0 and self.model.jnt_type[root_joint] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = self.model.jnt_qposadr[root_joint]
                self.data.qpos[qpos_adr + 2] += delta
            else:
                self.model.body_pos[root, 2] += delta

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _trajectory_anchor_indices(trajectory: dict, gripper_opening=None) -> list[int]:
        """Keep corners and opening transitions, dropping redundant samples."""
        names = list(trajectory)
        length = len(next(iter(trajectory.values())))
        anchors = {0, length - 1}
        for index in range(1, length - 1):
            for name in names:
                previous = index - 1
                while previous >= 0 and np.linalg.norm(
                    np.asarray(trajectory[name][index][0])
                    - np.asarray(trajectory[name][previous][0])
                ) <= 1e-12:
                    previous -= 1
                following = index + 1
                while following < length and np.linalg.norm(
                    np.asarray(trajectory[name][following][0])
                    - np.asarray(trajectory[name][index][0])
                ) <= 1e-12:
                    following += 1
                if previous < 0 or following >= length:
                    continue
                before = np.asarray(trajectory[name][index][0]) - np.asarray(
                    trajectory[name][previous][0]
                )
                after = np.asarray(trajectory[name][following][0]) - np.asarray(
                    trajectory[name][index][0]
                )
                before_norm, after_norm = np.linalg.norm(before), np.linalg.norm(after)
                if np.linalg.norm(before / before_norm - after / after_norm) > 1e-8:
                    anchors.add(index)
            if gripper_opening is not None:
                for name in names:
                    before = gripper_opening[name][index] - gripper_opening[name][index - 1]
                    after = gripper_opening[name][index + 1] - gripper_opening[name][index]
                    if abs(before - after) > 1e-8:
                        anchors.add(index)
        return sorted(anchors)

    def _retime_trajectories(
        self,
        interactor_trajectory: dict,
        max_controller_speed: float,
        free_controller_speed: float,
        substeps: int,
        gripper_opening: dict[str, list[float]] | None,
        slowdown_epsilon: float,
        object_body: str,
    ) -> tuple[dict, dict[str, list[float]] | None]:
        """Retime straight path segments without changing their geometry."""
        names = list(interactor_trajectory)
        lower, upper = self._object_bounds(object_body)
        anchors = self._trajectory_anchor_indices(interactor_trajectory, gripper_opening)
        retimed = {name: [interactor_trajectory[name][0]] for name in names}
        retimed_opening = (
            {name: [gripper_opening[name][0]] for name in names}
            if gripper_opening
            else None
        )

        for start_index, end_index in zip(anchors[:-1], anchors[1:]):
            starts = {
                name: np.asarray(interactor_trajectory[name][start_index][0], dtype=float)
                for name in names
            }
            targets = {
                name: np.asarray(interactor_trajectory[name][end_index][0], dtype=float)
                for name in names
            }
            intervals = [
                self._segment_bounds_interval(
                    starts[name], targets[name], lower, upper, slowdown_epsilon
                )
                for name in names
            ]
            breakpoints = {0.0, 1.0}
            for interval in intervals:
                if interval is not None:
                    breakpoints.update(interval)
            breakpoints = sorted(breakpoints)
            for segment_start, segment_end in zip(breakpoints[:-1], breakpoints[1:]):
                if segment_end - segment_start <= 1e-12:
                    continue
                near_object = any(
                    interval is not None
                    and interval[0] <= (segment_start + segment_end) * 0.5 <= interval[1]
                    for interval in intervals
                )
                speed = max_controller_speed if near_object else free_controller_speed
                max_distance = speed * self.model.opt.timestep * substeps
                segment_steps = max(
                    1,
                    max(
                        int(
                            np.ceil(
                                np.linalg.norm(targets[name] - starts[name])
                                * (segment_end - segment_start)
                                / max_distance
                            )
                        )
                        for name in names
                    ),
                )
                for step in range(1, segment_steps + 1):
                    fraction = segment_start + (
                        segment_end - segment_start
                    ) * step / segment_steps
                    for name in names:
                        position = starts[name] + fraction * (targets[name] - starts[name])
                        retimed[name].append(
                            (tuple(position), interactor_trajectory[name][end_index][1])
                        )
                        if retimed_opening is not None:
                            opening = (
                                gripper_opening[name][start_index]
                                + fraction
                                * (
                                    gripper_opening[name][end_index]
                                    - gripper_opening[name][start_index]
                                )
                            )
                            retimed_opening[name].append(opening)

        return retimed, retimed_opening

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
        controller_free_speed: float = 1.0,
        controller_slowdown_epsilon: float = 0.12,
        controller_object_body: str = "object",
    ) -> list[Frame]:
        """Play a scripted trajectory and record synthetic RGB-D + ground truth.

        ``max_controller_speed`` retimes only segments within
        ``controller_slowdown_epsilon`` metres of ``controller_object_body``.
        The original geometric path is preserved; additional samples are
        inserted only near the object.
        """
        if capture_every < 1:
            raise ValueError("capture_every must be at least 1")
        if substeps < 1:
            raise ValueError("substeps must be at least 1")
        if max_controller_speed is not None and max_controller_speed <= 0.0:
            raise ValueError("max_controller_speed must be positive")
        if controller_free_speed <= 0.0:
            raise ValueError("controller_free_speed must be positive")
        if controller_slowdown_epsilon < 0.0:
            raise ValueError("controller_slowdown_epsilon cannot be negative")
        if not interactor_trajectory:
            raise ValueError("at least one interactor trajectory is required")
        if max_controller_speed is not None:
            interactor_trajectory, gripper_opening = self._retime_trajectories(
                interactor_trajectory,
                max_controller_speed,
                controller_free_speed,
                substeps,
                gripper_opening,
                controller_slowdown_epsilon,
                controller_object_body,
            )
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
                pos, quat = interactor_trajectory[name][t]
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