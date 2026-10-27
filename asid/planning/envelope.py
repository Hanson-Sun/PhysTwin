"""The feasible set of tool motions: workspace, envelope, slew and contact.

One owner of every constraint a candidate must satisfy, and of the box the
search draws from. A candidate that leaves the set is infeasible rather than
penalised, because the information score is exactly flat before contact: a claw
that misses the object ties with the nominal plan and would otherwise dominate
the elites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asid.planning import spline
from asid.planning.actions import minimum_clearance
from asid.planning.spline import Motion, Spec


def within_workspace(
    controller: np.ndarray,
    contact: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_speed: float,
    dt: float,
) -> bool:
    """True if both clouds stay inside the box and below the framewise speed."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
        or not np.isfinite(max_speed)
        or not np.isfinite(dt)
        or max_speed <= 0
        or dt <= 0
    ):
        raise ValueError("invalid workspace, speed, or timestep")
    for points in (controller, contact):
        if np.any(points < lower) or np.any(points > upper):
            return False
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=-1) > max_speed * dt):
            return False
    return True


@dataclass(frozen=True)
class Diagnostics:
    """The numbers a plan is worth reporting: how hard the constraints bind."""

    max_offset: float
    max_yaw: float
    max_pitch: float
    max_slew: float
    min_clearance: float

    def summary(self) -> str:
        return (
            f"max offset {self.max_offset:.3f} m, yaw {self.max_yaw:.2f} rad, "
            f"pitch {self.max_pitch:.2f} rad, slew {self.max_slew:.3f} rad/frame, "
            f"clearance {self.min_clearance:.3f} m"
        )


@dataclass(frozen=True)
class Envelope:
    """Box, speed, orientation, slew and contact limits for one plan."""

    lower: np.ndarray
    upper: np.ndarray
    max_speed: float
    offset_limit: float
    orientation_limit: float
    max_rate: float
    dt: float
    contact_radius: float
    nodes: np.ndarray
    frames: int

    @classmethod
    def from_demo(
        cls,
        nodes: np.ndarray,
        template: tuple[np.ndarray, np.ndarray],
        *,
        frames: int,
        dt: float,
        contact_radius: float,
        margin: float = 0.15,
        offset_limit: float = 0.05,
        orientation_limit: float = spline.ORIENTATION_LIMIT,
        speed_factor: float = 1.5,
        max_rate: float = spline.MAX_ANGULAR_RATE,
    ) -> "Envelope":
        """Wrap what the demonstration occupies, with room to move and to be faster.

        The workspace is the object plus the recorded tool path, padded: an
        object-only box would reject the demonstration's own approach, which
        starts half a stroke away.
        """
        controller, contact = (np.asarray(cloud, dtype=np.float64) for cloud in template)
        occupied = np.concatenate(
            [np.asarray(nodes, dtype=np.float64), controller.reshape(-1, 3), contact.reshape(-1, 3)]
        )
        peak = max(
            float(np.linalg.norm(np.diff(cloud, axis=0), axis=-1).max())
            for cloud in (controller, contact)
        )
        return cls(
            lower=occupied.min(axis=0) - margin,
            upper=occupied.max(axis=0) + margin,
            max_speed=speed_factor * peak / dt,
            offset_limit=offset_limit,
            orientation_limit=orientation_limit,
            max_rate=max_rate,
            dt=dt,
            contact_radius=contact_radius,
            nodes=np.asarray(nodes, dtype=np.float64),
            frames=frames,
        )

    def bounds(self, knots: int) -> tuple[np.ndarray, np.ndarray]:
        """Box over :func:`spline.flatten`, with the first waypoint pinned.

        The tool is already at the recorded start pose, so a plan must not begin
        somewhere else: the first knot's offset *and* its yaw and pitch are fixed
        at zero, otherwise the replay's first command carries a wrist step
        (measured 15.5° / 9.8° at 124 frames before this was pinned). Rotation is
        still free at every later waypoint.
        """
        if knots < 2 or not np.isfinite(self.offset_limit) or self.offset_limit <= 0.0:
            raise ValueError("invalid knot count or offset limit")
        free_yaw = np.concatenate([np.zeros(1), np.full(knots - 1, -self.orientation_limit)])
        free_yaw_upper = np.concatenate([np.zeros(1), np.full(knots - 1, self.orientation_limit)])
        lower = np.concatenate(
            [
                np.concatenate([np.zeros(3), np.full(3 * knots - 3, -self.offset_limit)]),
                np.full(knots - 2, spline.MIN_TIME_GAP),
                free_yaw,
                free_yaw,
            ]
        )
        upper = np.concatenate(
            [
                np.concatenate([np.zeros(3), np.full(3 * knots - 3, self.offset_limit)]),
                np.full(knots - 2, 1.0 - spline.MIN_TIME_GAP),
                free_yaw_upper,
                free_yaw_upper,
            ]
        )
        return lower, upper

    def admit(
        self, spec: Spec, template: tuple[np.ndarray, np.ndarray]
    ) -> Motion | None:
        """The sampled motion, or ``None`` if any guard rejects it.

        The envelope is checked on the **sampled** frames as well as at the
        knots: with a monotone interpolant the two agree, and an interpolant that
        overshoots would be caught rather than silently commanding a wider motion
        than the box.
        """
        motion = spline.motion(spec, template[0], template[1], self.frames)
        if np.abs(motion.positions).max() > self.offset_limit:
            return None
        if max(np.abs(motion.yaw).max(), np.abs(motion.pitch).max()) > self.orientation_limit:
            return None
        if self._slew(motion) > self.max_rate * self.dt:
            return None
        if not within_workspace(
            motion.controller, motion.contact, self.lower, self.upper, self.max_speed, self.dt
        ):
            return None
        if minimum_clearance(self.nodes, motion.contact).min() > self.contact_radius:
            return None
        return motion

    def diagnostics(self, motion: Motion) -> Diagnostics:
        return Diagnostics(
            max_offset=float(np.abs(motion.positions).max()),
            max_yaw=float(np.abs(motion.yaw).max()),
            max_pitch=float(np.abs(motion.pitch).max()),
            max_slew=self._slew(motion),
            min_clearance=float(minimum_clearance(self.nodes, motion.contact).min()),
        )

    @staticmethod
    def _slew(motion: Motion) -> float:
        """Largest framewise wrist rotation, in radians."""
        return float(
            max(
                np.abs(np.diff(motion.yaw)).max(initial=0.0),
                np.abs(np.diff(motion.pitch)).max(initial=0.0),
            )
        )
