"""Carry a gripped object with the claw, without teleporting its state.

Closing the fingers is always physical (see ``DigitalTwinSim.set_gripper_opening``).
This module implements what happens *after* the grip closes -- the one part that
friction cannot be relied on for in a data-generation sim. Both mechanisms
capture their claw-relative pose exactly once, at the grasp, so engaging the
attach never teleports the object:

- ``WeldGrasp`` (rigid bodies): a MuJoCo ``weld`` equality between the claw and
  the object body, enabled at grasp time. Gravity, floor contact and the grip
  geometry keep acting; the object hangs from the grip in the pose it had when
  the grip closed.
- ``CarryGrasp`` (flex soft bodies): MuJoCo refuses a freejointed flex under the
  discrete integrator a flex needs, so there is no body to weld to. The flex's
  root frame is instead driven along the claw (position and orientation). The
  mesh keeps its own state and sags and deforms relative to that frame.
"""

from __future__ import annotations

import mujoco
import numpy as np


def flex_root_body(model: mujoco.MjModel, flex_id: int) -> int:
    """Return the body whose frame carries a flex's deformable mesh.

    A full-dof flex owns a body per vertex, a reduced-dof flex one per
    interpolation node; both sets descend from the flexcomp's own body, whose
    frame every vertex pose is composed with.
    """
    start, count = model.flex_vertadr[flex_id], model.flex_vertnum[flex_id]
    node, node_count = model.flex_nodeadr[flex_id], model.flex_nodenum[flex_id]
    bodies = np.concatenate(
        (
            model.flex_vertbodyid[start : start + count],
            model.flex_nodebodyid[node : node + node_count],
        )
    )
    owners = bodies[bodies >= 0]
    if not len(owners):
        raise ValueError("flex has no vertex bodies")
    return int(model.body_parentid[owners[0]])


def _relative_pose(claw_pos, claw_quat, pos, quat):
    """Return ``(pos, quat)`` of a body expressed in the claw's frame."""
    offset = np.asarray(pos, dtype=float) - np.asarray(claw_pos, dtype=float)
    inverse = np.empty(4)
    mujoco.mju_negQuat(inverse, np.asarray(claw_quat, dtype=float))
    rel_pos = np.empty(3)
    mujoco.mju_rotVecQuat(rel_pos, offset, inverse)
    rel_quat = np.empty(4)
    mujoco.mju_mulQuat(rel_quat, inverse, np.asarray(quat, dtype=float))
    return rel_pos, rel_quat


def _compose_pose(claw_pos, claw_quat, rel_pos, rel_quat):
    """Return the world pose of ``rel_pose`` carried by the claw."""
    pos = np.empty(3)
    mujoco.mju_rotVecQuat(
        pos, np.asarray(rel_pos, dtype=float), np.asarray(claw_quat, dtype=float)
    )
    pos += np.asarray(claw_pos, dtype=float)
    quat = np.empty(4)
    mujoco.mju_mulQuat(
        quat, np.asarray(claw_quat, dtype=float), np.asarray(rel_quat, dtype=float)
    )
    return pos, quat


class GraspAttachment:
    """Keeps a gripped body with the claw once the grip has closed.

    Created for a ``(body, claw)`` pair; ``activate`` captures the relative
    pose at the grasp and ``follow`` keeps a soft body's frame on the claw
    afterwards (rigid bodies are carried by the weld alone).
    """

    def __init__(self, model: mujoco.MjModel, body_name: str, claw_name: str):
        self.body_name = body_name
        self.claw_name = claw_name
        self.active = False
        self._rel_pos = np.zeros(3)
        self._rel_quat = np.array((1.0, 0.0, 0.0, 0.0))

        flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, body_name)
        if flex_id >= 0:
            self.flex_id = flex_id
            self.root_body = flex_root_body(model, flex_id)
            self.weld_id = -1
        else:
            self.flex_id = -1
            joint_id = model.body(body_name).jntadr[0]
            if (
                joint_id < 0
                or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
            ):
                raise ValueError(
                    f"grasped body '{body_name}' must have a freejoint or be a flex"
                )
            self.root_body = model.body(body_name).id
            self.weld_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{claw_name}_grasp_weld"
            )
            if self.weld_id < 0:
                raise ValueError(f"scene has no weld equality for '{claw_name}'")
        self.claw_id = model.body(claw_name).id

    def _claw_pose(self, data: mujoco.MjData):
        """The claw's world pose, read from the kinematics results.

        ``xpos``/``xquat`` hold the resolved pose for *any* body. For a mocap
        claw they are identical to ``mocap_pos``/``mocap_quat`` (mocap drives
        kinematics directly), so this reads the same numbers on the kinematic
        path while also working when the claw is a servo-driven finite-mass
        body -- where the pose lags the command and only the resolved value is
        the true grip pose.
        """
        return data.xpos[self.claw_id], data.xquat[self.claw_id]

    def activate(
        self, model: mujoco.MjModel, data: mujoco.MjData, offset=(0.0, 0.0, 0.0)
    ) -> None:
        """Capture the claw-relative pose at the grasp and engage the attach.

        ``offset`` shifts the carried pose in world axes, for callers that want
        the object held slightly off the grip centre.
        """
        claw_pos, claw_quat = self._claw_pose(data)
        body = model.body(self.body_name).id
        rel_pos, rel_quat = _relative_pose(
            claw_pos, claw_quat, data.xpos[body], data.xquat[body]
        )
        offset = np.asarray(offset, dtype=float)
        if np.any(offset):
            inverse = np.empty(4)
            mujoco.mju_negQuat(inverse, np.asarray(claw_quat, dtype=float))
            shift = np.empty(3)
            mujoco.mju_rotVecQuat(shift, np.asarray(offset, dtype=float), inverse)
            rel_pos = rel_pos + shift
        self._rel_pos, self._rel_quat = rel_pos, rel_quat
        if self.weld_id >= 0:
            model.eq_data[self.weld_id, 3:6] = rel_pos
            model.eq_data[self.weld_id, 6:10] = rel_quat
            data.eq_active[self.weld_id] = True
        self.active = True

    def follow(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Keep a carried soft body's frame on the claw (no-op for welds)."""
        if not self.active or self.weld_id >= 0:
            return
        claw_pos, claw_quat = self._claw_pose(data)
        pos, quat = _compose_pose(
            claw_pos, claw_quat, self._rel_pos, self._rel_quat
        )
        model.body_pos[self.root_body] = pos
        model.body_quat[self.root_body] = quat
