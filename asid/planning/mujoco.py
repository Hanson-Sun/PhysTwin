"""Execute a translation-only planned controller trajectory in MuJoCo.

The planner works in PhysTwin's reflected world coordinates (z down). The
caller supplies the MuJoCo pose corresponding to the template's first frame;
controller point displacements must be rigid translations and quaternion is
held fixed. Timing is explicit rather than inferred from recorded tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asid.planning.trajectory import Trajectory


@dataclass(frozen=True)
class Execution:
    frames: list
    commands: np.ndarray
    times: np.ndarray


def execute(
    sim,
    trajectory: Trajectory,
    *,
    initial_position: np.ndarray,
    quaternion: np.ndarray,
    frame_dt: float,
    interactor: str = "interactor0",
    tolerance: float = 1e-4,
) -> Execution:
    """Execute planned point translations at fixed frame intervals.

    A mocap-driven interactor is required so commanded and actual poses agree.
    The current MuJoCo pose must match the provided first-frame alignment;
    this routine never silently teleports a controller to a trajectory start.
    The return contains the actual command per physics step and RGB-D frames,
    including the initial frame before the first physics step.
    """
    points = np.asarray(trajectory.controller, dtype=np.float64)
    contact = np.asarray(trajectory.contact, dtype=np.float64)
    initial = np.asarray(initial_position, dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or len(points) < 2 or points.shape[1] == 0:
        raise ValueError("controller must contain at least two nonempty 3D frames")
    if contact.ndim != 3 or contact.shape[0] != len(points) or contact.shape[1] == 0 or contact.shape[2] != 3:
        raise ValueError("contact points must match controller frames")
    if initial.shape != (3,) or quat.shape != (4,) or not all(np.isfinite(x).all() for x in (points, contact, initial, quat)):
        raise ValueError("trajectory and initial pose must be finite")
    if (not np.isfinite(frame_dt) or not np.isfinite(tolerance)
        or abs(np.linalg.norm(quat) - 1.0) > 1e-6 or frame_dt <= 0 or tolerance <= 0):
        raise ValueError("quaternion must be normalized and times/tolerance positive")
    displacement = points - points[0]
    translation = displacement.mean(axis=1)
    if np.max(np.abs(displacement - translation[:, None, :])) > tolerance:
        raise ValueError("controller trajectory is not a rigid translation")
    if np.max(np.abs((contact - contact[0]) - translation[:, None, :])) > tolerance:
        raise ValueError("contact points do not follow the controller translation")
    if sim.model.body(interactor).mocapid[0] < 0:
        raise ValueError("trajectory execution currently requires a mocap interactor")
    if np.linalg.norm(sim.commanded_interactor_pose(interactor) - initial) > tolerance:
        raise ValueError("current interactor position does not match initial alignment")
    mocap_id = sim.model.body(interactor).mocapid[0]
    current_quat = np.asarray(sim.data.mocap_quat[mocap_id], dtype=float)
    if min(np.linalg.norm(current_quat - quat), np.linalg.norm(current_quat + quat)) > tolerance:
        raise ValueError("current interactor orientation does not match initial alignment")

    physics_dt = float(sim.model.opt.timestep)
    steps = int(round(frame_dt / physics_dt))
    if steps < 1 or not np.isclose(steps * physics_dt, frame_dt, atol=1e-9, rtol=0):
        raise ValueError("frame_dt must be an integer multiple of the MuJoCo timestep")
    # Reflection between PhysTwin's z-down world and MuJoCo's z-up world.
    positions = initial + translation * np.array([1.0, 1.0, -1.0])
    from mujoco_sim.simulation import Frame

    start_time = float(sim.data.time)
    frames = [Frame(time=start_time, rgbd=sim.render_all_cameras(), object_qpos={})]
    commands, times = [], []
    for index in range(1, len(positions)):
        for substep in range(1, steps + 1):
            fraction = substep / steps
            position = positions[index - 1] + fraction * (positions[index] - positions[index - 1])
            sim.set_interactor_pose(interactor, position, quat)
            sim.step()
            commands.append(position.copy())
            times.append(float(sim.data.time) - start_time)
        frames.append(Frame(time=float(sim.data.time), rgbd=sim.render_all_cameras(), object_qpos={}))
    return Execution(frames, np.asarray(commands), np.asarray(times))
