"""Candidate pushes as rigid rotations of the recorded controller trajectory.

Each action is a pitch about the horizontal axis followed by a rotation about
the vertical axis through the object centroid, both about the same center.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from asid.fim.case import Case


@dataclass(frozen=True)
class Action:
    name: str
    azimuth: float
    controller: np.ndarray
    contact: np.ndarray
    tilt: float = 0.0


def _pitch(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _yaw(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def rotate_action(
    points: np.ndarray,
    azimuth: float,
    tilt: float,
    center: np.ndarray,
) -> np.ndarray:
    """Pitch by ``tilt`` about y, then yaw by ``azimuth`` about z, about ``center``."""
    if azimuth == 0.0 and tilt == 0.0:
        return np.array(points, dtype=np.float32, copy=True)
    rotation = _yaw(azimuth) @ _pitch(tilt)
    centered = np.asarray(points, dtype=np.float64) - center
    return (centered @ rotation.T + center).astype(np.float32)


def minimum_clearance(nodes: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Closest object point to the controller surface, per frame."""
    tree = cKDTree(np.asarray(nodes, dtype=np.float64))
    points = np.asarray(contact, dtype=np.float64)
    return np.array([tree.query(frame, k=1)[0].min() for frame in points])


def contact_frames(nodes: np.ndarray, contact: np.ndarray, radius: float) -> int:
    return int((minimum_clearance(nodes, contact) <= radius).sum())


def action_name(azimuth_degrees: float, tilt_degrees: float) -> str:
    azimuth = int(round(float(azimuth_degrees))) % 360
    if float(tilt_degrees) == 0.0:
        return f"az{azimuth:03d}"
    return f"az{azimuth:03d}_t{int(round(float(tilt_degrees))):+03d}"


def generate_actions(
    case: Case, azimuths, tilts=(0.0,)  # noqa: ANN001 - array-like of degrees
) -> list[Action]:
    """Every ``(azimuth, tilt)`` pair, rigid copies of the recorded push."""
    center = np.asarray(case.nodes, dtype=np.float64).mean(axis=0)
    actions = []
    for degrees in np.atleast_1d(azimuths):
        for tilt_degrees in np.atleast_1d(tilts):
            azimuth = float(np.deg2rad(float(degrees)))
            tilt = float(np.deg2rad(float(tilt_degrees)))
            actions.append(
                Action(
                    name=action_name(float(degrees), float(tilt_degrees)),
                    azimuth=azimuth,
                    controller=rotate_action(case.controller, azimuth, tilt, center),
                    contact=rotate_action(case.contact, azimuth, tilt, center),
                    tilt=tilt,
                )
            )
    return actions


def require_contact(
    case: Case, actions: list[Action], radius: float
) -> dict[str, float]:
    """Every action must reach the object; returns the minimum clearance per action."""
    clearance = {
        action.name: float(minimum_clearance(case.nodes, action.contact).min())
        for action in actions
    }
    misses = [name for name, value in clearance.items() if value > radius]
    if misses:
        raise ValueError(f"actions never contact the object within {radius} m: {misses}")
    return clearance
