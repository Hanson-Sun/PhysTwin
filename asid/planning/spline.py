"""Spline tool motions in PhysTwin's z-down world frame.A plan is ``knots`` waypoints, each with a position offset from the recorded
   tool path, a time fraction and a claw orientation. Position and angles follow a
   shape-preserving (PCHIP) spline in normalized time, so the waypoints bound the
   commanded motion, and the fractions are that spline's knot locations, so their
   spacing is the speed profile: the claw can dwell, press, back off and
   re-approach instead of sliding along one straight stroke.

The tool rotates rigidly about its own centre every frame, so the controller and
contact clouds keep their shape. Offsets are relative to the recorded path, so
zeros reproduce it and the plan can be replayed on a differently posed tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from asid.planning.actions import minimum_clearance
from asid.planning.cem import CemResult, cem
from asid.planning.trajectory import feasible_points

MIN_TIME_GAP = 1e-3
ORIENTATION_LIMIT = np.deg2rad(30.0)
# Wrist slew the plan may ask for; a poke does not spin the tool 90 deg/s.
MAX_ANGULAR_RATE = 1.5
REFLECTION = np.array([1.0, 1.0, -1.0])


@dataclass(frozen=True)
class Spec:
    """Waypoint description of one tool motion."""

    positions: np.ndarray  # (knots, 3) offset from the recorded tool path
    times: np.ndarray      # (knots,) ascending, first 0.0 and last 1.0
    yaw: np.ndarray        # (knots,) radians about z
    pitch: np.ndarray      # (knots,) radians about y

    @classmethod
    def start(cls, knots: int) -> "Spec":
        if knots < 2:
            raise ValueError("a trajectory needs at least two waypoints")
        return cls(
            positions=np.zeros((knots, 3)),
            times=np.linspace(0.0, 1.0, knots),
            yaw=np.zeros(knots),
            pitch=np.zeros(knots),
        )

    @property
    def knots(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class Plan:
    spec: Spec
    controller: np.ndarray
    contact: np.ndarray
    result: CemResult


def ordered_times(internal: Sequence[float], knots: int) -> np.ndarray:
    """Ascending fractions from unconstrained interior values."""
    if knots <= 2:
        return np.array([0.0, 1.0])
    times = np.concatenate([[0.0], np.sort(np.asarray(internal, dtype=np.float64)), [1.0]])
    times[1:-1] = np.clip(times[1:-1], MIN_TIME_GAP, 1.0 - MIN_TIME_GAP)
    if np.any(np.diff(times) < MIN_TIME_GAP):
        return np.linspace(0.0, 1.0, knots)
    return times


def flatten(spec: Spec) -> np.ndarray:
    """Vector form used by the search: offsets, interior times, yaw, pitch."""
    return np.concatenate(
        [spec.positions.ravel(), spec.times[1:-1], spec.yaw, spec.pitch]
    )


def unflatten(vector: np.ndarray, knots: int) -> Spec:
    vector = np.asarray(vector, dtype=np.float64)
    size = 6 * knots - 2
    if vector.shape != (size,):
        raise ValueError(f"expected {size} controls for {knots} knots, got {vector.shape}")
    positions = vector[: 3 * knots].reshape(knots, 3)
    times = ordered_times(vector[3 * knots : 4 * knots - 2], knots)
    yaw = vector[4 * knots - 2 : 5 * knots - 2]
    pitch = vector[5 * knots - 2 :]
    return Spec(positions, times, yaw, pitch)


def bounds(
    knots: int, offset_limit: float, orientation_limit: float = ORIENTATION_LIMIT
) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds for the search, with the first waypoint pinned to the start pose.

    The tool is already at the recorded start pose, so a plan must not begin
    somewhere else: the first knot's offset *and* its yaw and pitch are fixed at
    zero, otherwise the replay's first command carries a wrist step (measured
    15.5 deg / 9.8 deg at 124 frames before this was pinned). Rotation is still
    free at every later waypoint.
    """
    if knots < 2 or not np.isfinite(offset_limit) or offset_limit <= 0.0:
        raise ValueError("invalid knot count or offset limit")
    pinned_yaw = np.concatenate([np.zeros(1), np.full(knots - 1, -orientation_limit)])
    pinned_yaw_upper = np.concatenate([np.zeros(1), np.full(knots - 1, orientation_limit)])
    lower = np.concatenate(
        [
            np.concatenate([np.zeros(3), np.full(3 * knots - 3, -offset_limit)]),
            np.full(knots - 2, MIN_TIME_GAP),
            pinned_yaw,
            pinned_yaw,
        ]
    )
    upper = np.concatenate(
        [
            np.concatenate([np.zeros(3), np.full(3 * knots - 3, offset_limit)]),
            np.full(knots - 2, 1.0 - MIN_TIME_GAP),
            pinned_yaw_upper,
            pinned_yaw_upper,
        ]
    )
    return lower, upper


def _rotations(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Framewise ``Rz(yaw) @ Ry(pitch)``."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    zero, one = np.zeros_like(yaw), np.ones_like(yaw)
    rows_z = [
        np.stack([cy, -sy, zero], axis=-1),
        np.stack([sy, cy, zero], axis=-1),
        np.stack([zero, zero, one], axis=-1),
    ]
    rows_y = [
        np.stack([cp, zero, sp], axis=-1),
        np.stack([zero, one, zero], axis=-1),
        np.stack([-sp, zero, cp], axis=-1),
    ]
    return np.stack(rows_z, axis=-2) @ np.stack(rows_y, axis=-2)


def _rigid(points: np.ndarray, positions, yaw, pitch) -> np.ndarray:
    centre = np.asarray(points, dtype=np.float64).mean(axis=1, keepdims=True)
    local = np.asarray(points, dtype=np.float64) - centre
    rotated = np.einsum("fij,fpj->fpi", _rotations(yaw, pitch), local)
    return rotated + centre + positions[:, None, :]


def controls(spec: Spec, frames: int):
    """Per-frame offsets and angles of a spec, for replay outside the planner.

    Shape-preserving interpolation (PCHIP), not a natural cubic: a cubic
    overshoots between its knots, so the waypoint box would not bound the motion
    that is actually commanded (measured 0.25 m of offset and 53° of yaw from a
    box of 0.05 m and 30°). With PCHIP the waypoints are the extremes, so the box
    and the sampled motion agree and a bounded search stays feasible.
    """
    u = np.linspace(0.0, 1.0, frames)
    return (
        PchipInterpolator(spec.times, spec.positions, axis=0)(u),
        PchipInterpolator(spec.times, spec.yaw)(u),
        PchipInterpolator(spec.times, spec.pitch)(u),
    )


def sample(spec: Spec, controller: np.ndarray, contact: np.ndarray, frames: int):
    """Sample the tool path and the cloud motion over ``frames`` frames."""
    controller = np.asarray(controller, dtype=np.float64)
    contact = np.asarray(contact, dtype=np.float64)
    if len(controller) != len(contact):
        raise ValueError("controller and contact trajectories disagree")
    if not 2 <= frames <= len(controller):
        raise ValueError("frames must fit inside the recorded trajectory")
    positions, yaw, pitch = controls(spec, frames)
    return (
        _rigid(controller[:frames], positions, yaw, pitch),
        _rigid(contact[:frames], positions, yaw, pitch),
    )


def mirror(positions: np.ndarray, yaw: np.ndarray, pitch: np.ndarray):
    """PhysTwin (z down) to MuJoCo (z up): reflect z, keep yaw, negate pitch.

    ``M = diag(1, 1, -1)`` gives ``M Rz M = Rz`` and ``M Ry(theta) M = Ry(-theta)``.
    """
    return np.asarray(positions) * REFLECTION, np.asarray(yaw), -np.asarray(pitch)


def plan_spline(
    score: Callable[[np.ndarray, np.ndarray], float],
    template: tuple[np.ndarray, np.ndarray],
    nodes: np.ndarray,
    *,
    frames: int,
    lower: np.ndarray,
    upper: np.ndarray,
    max_speed: float,
    dt: float,
    contact_radius: float,
    knots: int = 3,
    offset_limit: float = 0.05,
    samples: int = 8,
    iterations: int = 3,
    elites: int = 2,
    seed: int = 0,
    orientation_limit: float = ORIENTATION_LIMIT,
    max_rate: float = MAX_ANGULAR_RATE,
) -> Plan:
    """Search a spline tool motion with CEM.

    ``score`` maps a sampled ``(controller, contact)`` pair to an information
    gain and ``lower``/``upper`` bound the absolute tool points. Candidates that
    leave the workspace, leave the offset or orientation envelope *as sampled*,
    move faster than ``max_speed``, slew faster than ``max_rate``, or never reach
    the object are infeasible: the gain is exactly flat before contact, so those
    would otherwise tie with the nominal plan and dominate the elites. The
    envelope is checked on the sampled frames as well as at the knots: with a
    monotone interpolant the two agree, and an interpolant that overshoots would
    be caught rather than silently commanding a wider motion than the box.
    """
    controller, contact = template

    def objective(vector: np.ndarray) -> float | None:
        spec = unflatten(vector, knots)
        positions, yaw, pitch = controls(spec, frames)
        if np.abs(positions).max() > offset_limit:
            return None
        if max(np.abs(yaw).max(), np.abs(pitch).max()) > orientation_limit:
            return None
        if max(np.abs(np.diff(yaw)).max(), np.abs(np.diff(pitch)).max()) > max_rate * dt:
            return None
        sampled = sample(spec, controller, contact, frames)
        if not feasible_points(*sampled, lower, upper, max_speed, dt):
            return None
        if minimum_clearance(nodes, sampled[1]).min() > contact_radius:
            return None
        return float(score(*sampled))

    result = cem(
        objective,
        flatten(Spec.start(knots)),
        *bounds(knots, offset_limit, orientation_limit),
        samples=samples,
        iterations=iterations,
        elites=elites,
        project=lambda vector: flatten(unflatten(vector, knots)),
        seed=seed,
    )
    spec = unflatten(result.vector, knots)
    sampled = sample(spec, controller, contact, frames)
    return Plan(spec, sampled[0], sampled[1], result)
