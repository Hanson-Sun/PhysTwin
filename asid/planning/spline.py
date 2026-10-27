"""Spline tool motions in PhysTwin's z-down world frame.

A plan is ``knots`` waypoints, each with a position offset from the recorded
tool path, a time fraction and a claw orientation. Position and angles follow a
shape-preserving (PCHIP) spline in normalized time, so the waypoints bound the
commanded motion, and the fractions are that spline's knot locations, so their
spacing is the speed profile: the claw can dwell, press, back off and
re-approach instead of sliding along one straight stroke.

The tool rotates rigidly about its own centre every frame, so the controller and
contact clouds keep their shape. Offsets are relative to the recorded path, so
zeros reproduce it and the plan can be replayed on a differently posed tool.
This module is the parameterization only: the feasible set is
:mod:`asid.planning.envelope` and the search is :mod:`asid.planning.search`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

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
class Motion:
    """Per-frame tool clouds and the controls that produced them."""

    controller: np.ndarray
    contact: np.ndarray
    positions: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray


def ordered_times(internal, knots: int) -> np.ndarray:
    """Ascending fractions from unconstrained interior values."""
    if knots <= 2:
        return np.array([0.0, 1.0])
    times = np.concatenate(
        [[0.0], np.sort(np.asarray(internal, dtype=np.float64)), [1.0]]
    )
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


def motion(spec: Spec, controller: np.ndarray, contact: np.ndarray, frames: int) -> Motion:
    """Sample the tool path and the cloud motion over ``frames`` frames."""
    controller = np.asarray(controller, dtype=np.float64)
    contact = np.asarray(contact, dtype=np.float64)
    if len(controller) != len(contact):
        raise ValueError("controller and contact trajectories disagree")
    if not 2 <= frames <= len(controller):
        raise ValueError("frames must fit inside the recorded trajectory")
    positions, yaw, pitch = controls(spec, frames)
    return Motion(
        controller=_rigid(controller[:frames], positions, yaw, pitch),
        contact=_rigid(contact[:frames], positions, yaw, pitch),
        positions=positions,
        yaw=yaw,
        pitch=pitch,
    )


def mirror(positions: np.ndarray, yaw: np.ndarray, pitch: np.ndarray):
    """PhysTwin (z down) to MuJoCo (z up): reflect z, keep yaw, negate pitch.

    ``M = diag(1, 1, -1)`` gives ``M Rz M = Rz`` and ``M Ry(theta) M = Ry(-theta)``.
    """
    return np.asarray(positions) * REFLECTION, np.asarray(yaw), -np.asarray(pitch)
