"""Thin wrapper around a MuJoCo model for scripted interaction + RGB-D capture."""

from dataclasses import dataclass, field

import mujoco
import numpy as np

try:
    from .interactor import (
        controller_local_bounds,
        geom_local_half_extents,
        geom_world_extents,
        object_bounds,
        object_geom_ids,
    )
except ImportError:  # Support running this file directly from the mujoco_sim directory.
    from interactor import (
        controller_local_bounds,
        geom_local_half_extents,
        geom_world_extents,
        object_bounds,
        object_geom_ids,
    )


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
    # Hinge angle the fingers hold while open. 0.70 leaves a 0.20 m gap
    # between the pads; a wider object needs a wider stance to be approached
    # without the pads grazing it on the way down.
    gripper_open_angle: float = 0.70

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
        """Set one claw's normalized hinge closing target in [0, 1].

        ``closing`` interpolates the hinge between the angle that holds the
        fingers open (0) and the angle that grips the object (1). Each claw pad
        sits 0.052 m inboard of its hinge, so a hinge at 0.34 rad already puts
        both pads on a 0.10 m object's faces: a more closed target drives the
        pads through the object instead of around it, which squeezes the object
        out of the grip. The open stance defaults to 0.70 rad (0.20 m between
        the pads) and is set by ``gripper_open_angle``.
        """
        if not 0.0 <= closing <= 1.0:
            raise ValueError("gripper closing must be between 0 and 1")
        open_angle = self.gripper_open_angle
        closed_angle = 0.34
        left_motor = self.model.actuator(f"{name}_finger_l_motor").id
        right_motor = self.model.actuator(f"{name}_finger_r_motor").id
        # Both motors drive mirrored hinges, so their targets are opposite.
        travel = open_angle - closed_angle
        self.data.ctrl[left_motor] = open_angle - travel * closing
        self.data.ctrl[right_motor] = -open_angle + travel * closing

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

    def translate_flex(self, flex_id: int, delta) -> None:
        """Translate a whole flex by `delta`, in world coordinates.

        A flexcomp's bodies all descend from the body it was declared in, so
        shifting that body moves the deformable mesh rigidly. This replaces the
        freejoint write used to move a rigid object.
        """
        # A full-dof flex owns a body per vertex, a reduced-dof flex one per
        # interpolation node; both sets descend from the flexcomp's own body.
        start, count = self.model.flex_vertadr[flex_id], self.model.flex_vertnum[flex_id]
        node, node_count = self.model.flex_nodeadr[flex_id], self.model.flex_nodenum[flex_id]
        bodies = np.concatenate(
            (
                self.model.flex_vertbodyid[start : start + count],
                self.model.flex_nodebodyid[node : node + node_count],
            )
        )
        owners = bodies[bodies >= 0]
        if not len(owners):
            raise ValueError("flex has no bodies to translate")
        parent = int(self.model.body_parentid[owners[0]])
        self.model.body_pos[parent] += np.asarray(delta, dtype=float)

    def get_object_state(self, body_name: str) -> np.ndarray:
        """7-vec [pos(3), quat(4)] ground truth for a rigid or soft object.

        A soft object has no freejoint - its flex vertices carry the motion - so
        it is reported as the deformable mesh's centre of mass with an identity
        orientation. Rigid objects keep their free body pose.
        """
        flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, body_name)
        if flex_id < 0:
            return self.get_free_body_state(body_name)
        start = self.model.flex_vertadr[flex_id]
        count = self.model.flex_vertnum[flex_id]
        vertices = np.asarray(self.data.flexvert_xpos)[start : start + count]
        return np.concatenate([vertices.mean(axis=0), (1.0, 0.0, 0.0, 0.0)])

    # ---- data generation ------------------------------------------------

    def _body_geom_ids(self, body_name: str) -> list[int]:
        """Return geometry ids attached to a body or one of its descendants."""
        body_id = self.model.body(body_name).id
        geom_ids = []
        for geom_id in range(self.model.ngeom):
            ancestor = self.model.geom_bodyid[geom_id]
            while ancestor > 0 and ancestor != body_id:
                ancestor = self.model.body_parentid[ancestor]
            if ancestor == body_id:
                geom_ids.append(geom_id)
        return geom_ids

    def _object_bounds(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return world-axis-aligned bounds for the object's geometry."""
        return object_bounds(self.model, self.data, body_name)

    def _object_oriented_bounds(
        self, body_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (center, rotation, half extents) of the object's bounding box.

        A rigid object is bounded by a box attached to its body, so the box
        turns with the object instead of puffing up to the axis-aligned hull of
        its rotated geometry. The heavy-end plank is rotated 70 degrees about
        z: its axis-aligned hull reaches 0.099 m along x only at |y| = 0.16 m,
        where the claw travelling along y = 0 can never touch it, which used to
        hold the claw at ``max_controller_speed`` from ~0.15 m away.

        Objects whose geometry does not belong to the ``body_name`` body - flex
        ropes and top-level composite chains - cannot be bounded this way and
        keep the world-axis-aligned union bounds.
        """
        body_id = self.model.body(body_name).id
        owned = set(self._body_geom_ids(body_name))
        scene_geoms = set(object_geom_ids(self.model))
        if self.model.nflexvert or not scene_geoms or not scene_geoms <= owned:
            lower, upper = self._object_bounds(body_name)
            return (lower + upper) * 0.5, np.eye(3), (upper - lower) * 0.5

        rotation = self.data.xmat[body_id].reshape(3, 3)
        origin = self.data.xpos[body_id]
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for geom_id in owned:
            local_half = geom_local_half_extents(self.model, geom_id)
            # Express each geom's own bounding box in the object body frame.
            half = (
                np.abs(rotation.T @ self.data.geom_xmat[geom_id].reshape(3, 3))
                @ local_half
            )
            center = rotation.T @ (self.data.geom_xpos[geom_id] - origin)
            lower = np.minimum(lower, center - half)
            upper = np.maximum(upper, center + half)
        center = origin + rotation @ ((lower + upper) * 0.5)
        return center, rotation, (upper - lower) * 0.5

    def _controller_local_bounds(
        self, body_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return controller bounds relative to its mocap reference point."""
        return controller_local_bounds(self.model, self.data, body_name)

    def _controller_geom_boxes(
        self, body_name: str
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return ``(center, rotation, half extents)`` for each controller geom.

        ``center`` is relative to the controller's mocap reference point, so the
        boxes translate with the commanded pose. One box per geometry keeps the
        envelope tight: the claw's housing and palm sit well above a table-top
        object, and a single box around all of them used to keep the claw slow
        long before any part of it could touch the object.
        """
        mocap_id = self.model.body(body_name).mocapid[0]
        if mocap_id < 0:
            raise ValueError(f"'{body_name}' is not a mocap body")
        origin = self.data.mocap_pos[mocap_id]
        boxes = []
        for geom_id in self._body_geom_ids(body_name):
            boxes.append(
                (
                    self.data.geom_xpos[geom_id] - origin,
                    self.data.geom_xmat[geom_id].reshape(3, 3),
                    geom_local_half_extents(self.model, geom_id),
                )
            )
        if not boxes:
            raise ValueError(f"could not find geometry for controller '{body_name}'")
        return boxes

    @staticmethod
    def _box_proximity_interval(
        start: np.ndarray,
        end: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        object_half: np.ndarray,
        controller_center: np.ndarray,
        controller_rotation: np.ndarray,
        controller_half: np.ndarray,
        epsilon: float,
    ) -> tuple[float, float] | None:
        """Return where a segment of reference points puts two boxes in contact.

        The reference points that make the controller box touch the object box
        form the Minkowski sum of both boxes. Two boxes intersect exactly when
        no separating axis exists, and the candidate axes are the six face
        normals plus the nine edge cross products, so clipping the segment
        against those slabs is exact for any orientation.
        """
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        origin = (
            np.asarray(start, dtype=float)
            + np.asarray(controller_center, dtype=float)
            - np.asarray(object_center, dtype=float)
        )
        object_axes = [object_rotation[:, axis] for axis in range(3)]
        controller_axes = [controller_rotation[:, axis] for axis in range(3)]
        normals = object_axes + controller_axes
        for object_axis in object_axes:
            for controller_axis in controller_axes:
                normal = np.cross(object_axis, controller_axis)
                length = np.linalg.norm(normal)
                if length > 1e-9:
                    normals.append(normal / length)

        entry, exit = 0.0, 1.0
        for normal in normals:
            reach = (
                float(np.sum(object_half * np.abs(object_rotation.T @ normal)))
                + float(
                    np.sum(controller_half * np.abs(controller_rotation.T @ normal))
                )
                + epsilon
            )
            offset = float(normal @ origin)
            slope = float(normal @ direction)
            if abs(slope) < 1e-12:
                if abs(offset) > reach:
                    return None
                continue
            near = (-reach - offset) / slope
            far = (reach - offset) / slope
            if near > far:
                near, far = far, near
            entry, exit = max(entry, near), min(exit, far)
            if entry > exit:
                return None
        return entry, exit

    @classmethod
    def _segment_proximity_interval(
        cls,
        start: np.ndarray,
        end: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        object_half: np.ndarray,
        controller_boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        epsilon: float,
    ) -> tuple[float, float] | None:
        """Return the first and last parameter with any controller box in contact.

        The contact set along the segment is the union of the per-geometry
        intervals, so the entry is the earliest of them and the exit the latest.
        """
        entry = exit = None
        for controller_center, controller_rotation, controller_half in controller_boxes:
            interval = cls._box_proximity_interval(
                start,
                end,
                object_center,
                object_rotation,
                object_half,
                controller_center,
                controller_rotation,
                controller_half,
                epsilon,
            )
            if interval is None:
                continue
            entry = interval[0] if entry is None else min(entry, interval[0])
            exit = interval[1] if exit is None else max(exit, interval[1])
        if entry is None:
            return None
        return entry, exit

    def place_objects_on_ground(self, ground_z: float = 0.0) -> None:
        """Place every non-controller object root on the ground before frame 0.

        Soft objects are unaffected: a flex carries neither geometry nor a
        freejoint, and `soft_body.soft_object` already grounds the mesh it
        tetrahedralizes.
        """
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
                extents = geom_world_extents(self.model, self.data, geom_id)
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

    def _step_distance(
        self,
        current: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        names: list[str],
        max_controller_speed: float,
        controller_free_speed: float,
        substeps: int,
        slowdown_epsilon: float,
        object_body: str,
    ) -> float:
        """Return how far every controller may travel in one command, in metres.

        Controllers move at ``controller_free_speed`` while their geometry is
        clear of the object, at ``max_controller_speed`` once any of their boxes
        is inside the object's proximity box, and stop at the box boundary when
        they are about to enter it so it is never crossed at free speed.
        """
        object_center, object_rotation, object_half = self._object_oriented_bounds(
            object_body
        )
        free_distance = controller_free_speed * self.model.opt.timestep * substeps
        slow_distance = max_controller_speed * self.model.opt.timestep * substeps
        allowed_distance = free_distance

        for name in names:
            distance = float(
                np.linalg.norm(
                    np.asarray(targets[name], dtype=float)
                    - np.asarray(current[name], dtype=float)
                )
            )
            if distance <= 1e-12:
                continue
            interval = self._segment_proximity_interval(
                current[name],
                targets[name],
                object_center,
                object_rotation,
                object_half,
                self._controller_geom_boxes(name),
                slowdown_epsilon,
            )
            if interval is None:
                continue
            if interval[0] <= 1e-12:
                allowed_distance = min(allowed_distance, slow_distance)
            else:
                # Do not cross into the current proximity box at free speed.
                allowed_distance = min(allowed_distance, interval[0] * distance)

        return max(allowed_distance, 0.0)

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
        controller_slowdown_epsilon: float = 0.02,
        controller_object_body: str = "object",
    ) -> list[Frame]:
        """Play a scripted trajectory and record synthetic RGB-D + ground truth.

        ``max_controller_speed`` is applied online when the current controller
        geometry reaches the current object bounds, with the optional small
        ``controller_slowdown_epsilon`` safety margin. Bounds are recomputed
        after every physics step, so object motion affects later commands, and
        ``controller_free_speed`` is the speed used while the controller is
        clear of the object.

        ``grasped_body`` kinematically attaches that body from
        ``grasp_start_fraction`` onwards so the lift never depends on grip
        friction. The attach is anchored at the pose the body has when the
        grip closes (captured once as a claw-relative offset), so attaching
        never teleports the object; ``grasp_offset`` then shifts that target.
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
        if not 0.0 <= grasp_start_fraction <= 1.0:
            raise ValueError("grasp_start_fraction must be between 0 and 1")
        flex_id = None
        qpos_adr = qvel_adr = None
        if grasped_body is not None:
            flex_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_FLEX, grasped_body
            )
            if flex_id < 0:
                joint_id = self.model.body(grasped_body).jntadr[0]
                if (
                    joint_id < 0
                    or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
                ):
                    raise ValueError(
                        f"grasped body '{grasped_body}' must have a freejoint or be a flex"
                    )
                qpos_adr = self.model.jnt_qposadr[joint_id]
                qvel_adr = self.model.jnt_dofadr[joint_id]

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

        command_index = 0
        # Pose delta captured the first time the grasp attaches: the body keeps
        # the pose it had when the grip closed and follows the claw from there.
        attach_offset = None
        current_opening = (
            {
                name: float(gripper_opening[name][0])
                for name in names
            }
            if gripper_opening is not None
            else None
        )
        # One command spends its whole distance allowance, so it keeps moving
        # through waypoints that allowance reaches instead of advancing a single
        # waypoint per physics step. Consecutive waypoints inside a trajectory
        # segment are collinear, so the travelled path is unchanged and corners
        # are still hit exactly; without this the claw crawls at
        # ``waypoint spacing / (substeps * timestep)`` whenever the trajectory is
        # sampled finer than the allowed step. A waypoint that repeats the
        # previous pose still costs one command, so holds keep their duration and
        # the gripper interpolation keeps its resolution.
        waypoint = 0
        while waypoint < n_waypoints:
            command_waypoint = waypoint
            current = {
                name: self.data.mocap_pos[self.model.body(name).mocapid[0]].copy()
                for name in names
            }
            budget = None
            spent = 0.0
            clamped = False
            if max_controller_speed is not None:
                targets = {
                    name: np.asarray(
                        interactor_trajectory[name][command_waypoint][0], dtype=float
                    )
                    for name in names
                }
                budget = self._step_distance(
                    current,
                    targets,
                    names,
                    max_controller_speed,
                    controller_free_speed,
                    substeps,
                    controller_slowdown_epsilon,
                    controller_object_body,
                )

            while waypoint < n_waypoints:
                targets = {
                    name: np.asarray(
                        interactor_trajectory[name][waypoint][0], dtype=float
                    )
                    for name in names
                }
                remaining = max(
                    (
                        float(np.linalg.norm(targets[name] - current[name]))
                        for name in names
                    ),
                    default=0.0,
                )
                if remaining <= 1e-12:
                    # A repeated waypoint is a hold: the claw stays put, so the
                    # gripper keeps closing at its own waypoint rate. Applying
                    # the opening here - not only while travelling - is what
                    # spreads the grasp over the hold instead of snapping the
                    # fingers shut on the first moving command afterwards.
                    if current_opening is not None:
                        for name in names:
                            current_opening[name] = float(
                                gripper_opening[name][waypoint]
                            )
                            self.set_gripper_opening(name, current_opening[name])
                    waypoint += 1
                    break
                if budget is None:
                    step = remaining
                else:
                    step = min(budget - spent, remaining)
                    if step <= 1e-12:
                        break
                    # The allowance is per leg: the leg decides whether the claw
                    # may still travel at free speed.
                    allowed = self._step_distance(
                        current,
                        targets,
                        names,
                        max_controller_speed,
                        controller_free_speed,
                        substeps,
                        controller_slowdown_epsilon,
                        controller_object_body,
                    )
                    clamped = allowed < step
                    if clamped:
                        step = allowed

                fraction = step / remaining
                next_positions = {
                    name: current[name] + fraction * (targets[name] - current[name])
                    for name in names
                }
                for name in names:
                    self.set_interactor_pose(
                        name,
                        next_positions[name],
                        interactor_trajectory[name][waypoint][1],
                    )
                    if current_opening is not None:
                        current_opening[name] += fraction * (
                            gripper_opening[name][waypoint] - current_opening[name]
                        )
                        self.set_gripper_opening(name, current_opening[name])
                current = next_positions
                spent += step
                if step >= remaining - 1e-12:
                    waypoint += 1
                if budget is None or clamped:
                    break

            if grasped_body is not None and command_waypoint >= int(
                (n_waypoints - 1) * grasp_start_fraction
            ):
                claw_position = np.mean(
                    [
                        self.data.mocap_pos[self.model.body(name).mocapid[0]]
                        for name in names
                    ],
                    axis=0,
                )
                if attach_offset is None:
                    # Anchor the attach where the body already is. Snapping it
                    # to the claw's reference point instead teleported the
                    # object by the whole grip-time mismatch: the trajectory
                    # aims at the bounds/band centre while a soft body reports
                    # its vertex mean, and a squishy body sags on the floor and
                    # gets nudged by the closing pads before the grip closes.
                    # The offset is captured once, so the first attach command
                    # moves nothing and every later command only follows the
                    # claw (plus the residual physics drift).
                    attach_offset = (
                        self.get_object_state(grasped_body)[:3] - claw_position
                    )
                target = (
                    claw_position
                    + np.asarray(grasp_offset, dtype=float)
                    + attach_offset
                )
                if flex_id is not None:
                    self.translate_flex(
                        flex_id, target - self.get_object_state(grasped_body)[:3]
                    )
                else:
                    self.data.qpos[qpos_adr : qpos_adr + 3] = target
                    self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
                mujoco.mj_forward(self.model, self.data)

            self.step(substeps)

            if progress is not None:
                progress(1)

            if command_index % capture_every == 0:
                frames.append(
                    Frame(
                        time=self.data.time,
                        rgbd=self.render_all_cameras(),
                        object_qpos={
                            b: self.get_object_state(b) for b in object_bodies
                        },
                    )
                )
            command_index += 1

        return frames