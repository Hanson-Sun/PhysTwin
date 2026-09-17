"""Helpers to script interactor motions (pokes, presses, ...)."""

import mujoco
import numpy as np

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _lerp_segment(p0, p1, steps):
    if steps < 1:
        raise ValueError("steps must be at least 1")
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    if p0.shape != (3,) or p1.shape != (3,):
        raise ValueError("trajectory positions must be 3D coordinates")
    return [tuple(p0 + (p1 - p0) * t) for t in np.linspace(0, 1, steps)]


def poke_trajectory(
    start,
    target,
    approach_steps: int = 30,
    hold_steps: int = 20,
    retract_steps: int = 30,
    quat=IDENTITY_QUAT,
) -> list:
    """Waypoints for a single poke: approach -> hold at target -> retract."""
    if hold_steps < 0:
        raise ValueError("hold_steps cannot be negative")
    approach = _lerp_segment(start, target, approach_steps)
    hold = [tuple(np.asarray(target, dtype=float))] * hold_steps
    retract = _lerp_segment(target, start, retract_steps)
    return [(p, quat) for p in approach + hold + retract]


def multi_poke_trajectory(waypoints: list, steps_per_segment: int = 30, quat=IDENTITY_QUAT) -> list:
    """Interpolate a sequence of positions into scripted interactor poses."""
    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")
    trajectory = []
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        trajectory += _lerp_segment(p0, p1, steps_per_segment)
    return [(p, quat) for p in trajectory]


def object_bounds(model, data, body_name: str = "object") -> tuple[np.ndarray, np.ndarray]:
    """Return initial world-space bounds for an object body and its descendants."""
    body_id = model.body(body_name).id
    points = []

    if model.nflexvert:
        points.extend(data.flexvert_xpos)

    for geom_id in range(model.ngeom):
        ancestor = model.geom_bodyid[geom_id]
        while ancestor > 0 and ancestor != body_id:
            ancestor = model.body_parentid[ancestor]
        if ancestor == body_id:
            radius = model.geom_rbound[geom_id]
            points.append(data.geom_xpos[geom_id] - radius)
            points.append(data.geom_xpos[geom_id] + radius)

    if not points:
        raise ValueError(f"could not find geometry for body '{body_name}'")
    points = np.asarray(points, dtype=float)
    return points.min(axis=0), points.max(axis=0)


def object_push_trajectory(
    model,
    data,
    steps_per_segment: int = 30,
    body_name: str = "object",
    margin: float = 0.06,
    approach_height: float = 0.16,
) -> list:
    """Approach the object's left side, lower to its center, and push through it."""
    lower, upper = object_bounds(model, data, body_name)
    center = (lower + upper) * 0.5
    # Keep small floor-level objects (for example the rope) at their actual
    # center height instead of lifting the interactor above them.
    contact_z = max(0.012, center[2])
    side_x = lower[0] - margin
    exit_x = upper[0] + margin
    y = center[1]
    points = [
        (side_x, y, contact_z + approach_height),
        (side_x, y, contact_z),
        (exit_x, y, contact_z),
        (exit_x, y, contact_z + approach_height),
    ]
    return multi_poke_trajectory(points, steps_per_segment)
