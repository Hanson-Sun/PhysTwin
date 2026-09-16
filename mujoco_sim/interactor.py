"""Helpers to script interactor motions (pokes, presses, ...)."""

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
    """Waypoints for a single poke: approach -> hold at target -> retract.
    Returns a list of (pos, quat) tuples, one per simulation waypoint.
    """
    if hold_steps < 0:
        raise ValueError("hold_steps cannot be negative")
    approach = _lerp_segment(start, target, approach_steps)
    hold = [tuple(np.asarray(target, dtype=float))] * hold_steps
    retract = _lerp_segment(target, start, retract_steps)
    positions = approach + hold + retract
    return [(p, quat) for p in positions]


def multi_poke_trajectory(waypoints: list, steps_per_segment: int = 30, quat=IDENTITY_QUAT) -> list:
    """Waypoints for a sequence of positions, linearly interpolated between
    consecutive points (e.g. poke several spots on an object in one rollout)."""
    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")
    traj = []
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        traj += _lerp_segment(p0, p1, steps_per_segment)
    return [(p, quat) for p in traj]
