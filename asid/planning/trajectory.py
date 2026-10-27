"""Bounded sampled trajectory design with optional checked gradient refinement.

Control variables are translations of an existing controller/contact template
at equally spaced knots. The initial offset is fixed at zero. This module does
not claim global optimality or differentiability through Warp contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from asid.fim.operators import Observation, parameterized_run
from asid.fim.sketch import gain_against, randomized_sketch
from asid.planning.envelope import within_workspace


@dataclass(frozen=True)
class Trajectory:
    offsets: np.ndarray  # (frames, 3), in the Warp world frame
    controller: np.ndarray
    contact: np.ndarray


@dataclass(frozen=True)
class Plan:
    trajectory: Trajectory
    controls: np.ndarray
    score: float
    sampled_scores: np.ndarray
    gradient_status: str


def rigid_template(
    controller_points: np.ndarray, contact_points: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a translation-only template from initial points and reference poses.

    The initial point clouds and reference positions must share a world frame.
    No pose or controller origin is inferred from tracked surface points.
    """
    controller_points = np.asarray(controller_points, dtype=np.float64)
    contact_points = np.asarray(contact_points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must have shape (frames, 3), with at least two frames")
    if any(points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 for points in (controller_points, contact_points)):
        raise ValueError("initial point clouds must have shape (points, 3)")
    if not all(np.isfinite(x).all() for x in (controller_points, contact_points, positions)):
        raise ValueError("template values must be finite")
    translation = positions - positions[0]
    return controller_points[None] + translation[:, None], contact_points[None] + translation[:, None]


def make_trajectory(controller: np.ndarray, contact: np.ndarray, controls: np.ndarray) -> Trajectory:
    """Linearly interpolate bounded translation knots, starting at zero."""
    controller = np.asarray(controller, dtype=np.float64)
    contact = np.asarray(contact, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controller.ndim != 3 or contact.ndim != 3 or controller.shape[2] != 3 or contact.shape[2] != 3:
        raise ValueError("controller and contact must have shape (frames, points, 3)")
    if len(controller) != len(contact) or len(controller) < 2:
        raise ValueError("controller and contact need the same number of frames >= 2")
    if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 1:
        raise ValueError("controls must have shape (knots, 3)")
    if not all(np.isfinite(x).all() for x in (controller, contact, controls)):
        raise ValueError("trajectory values must be finite")
    knots = np.vstack([np.zeros(3), controls])
    times = np.linspace(0.0, 1.0, len(controller))
    knot_times = np.linspace(0.0, 1.0, len(knots))
    offsets = np.column_stack([np.interp(times, knot_times, knots[:, i]) for i in range(3)])
    return Trajectory(offsets, controller + offsets[:, None, :], contact + offsets[:, None, :])


def feasible_points(
    controller: np.ndarray,
    contact: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_speed: float,
    dt: float,
) -> bool:
    """Check point clouds against world bounds and framewise speed.

    The check itself lives with the rest of the feasible set
    (:func:`asid.planning.envelope.within_workspace`).
    """
    return within_workspace(controller, contact, lower, upper, max_speed, dt)


def feasible(trajectory: Trajectory, lower: np.ndarray, upper: np.ndarray, max_speed: float, dt: float) -> bool:
    """Check all template points against world bounds and framewise speed."""
    return feasible_points(
        trajectory.controller, trajectory.contact, lower, upper, max_speed, dt
    )


def fisher_score(
    rollout,
    partition,
    observation: Observation,
    trajectory: Trajectory,
    parameter_samples: Sequence[np.ndarray],
    weights: np.ndarray,
    prior_std: np.ndarray,
    current_factors: np.ndarray | None,
    rank: int,
    parameter_step: float,
    seed: int,
    initial=None,
) -> float:
    """Approximate expected local gain, conditioned on accumulated factors.

    ``current_factors`` are in prior-whitened coordinates: current precision
    is I + current_factors @ current_factors.T. Common sketch directions are
    used across actions and parameter samples for score comparisons.
    """
    weights = np.asarray(weights, dtype=np.float64)
    prior_std = np.asarray(prior_std, dtype=np.float64)
    samples = [np.asarray(phi, dtype=np.float64) for phi in parameter_samples]
    if not samples or weights.shape != (len(samples),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("parameter samples require finite nonnegative weights")
    if prior_std.ndim != 1 or not np.isfinite(prior_std).all() or np.any(prior_std <= 0):
        raise ValueError("prior standard deviations must be finite and positive")
    if any(phi.shape != prior_std.shape or not np.isfinite(phi).all() for phi in samples):
        raise ValueError("parameter samples must match the grouped prior")
    if current_factors is not None:
        current_factors = np.asarray(current_factors, dtype=np.float64)
        if current_factors.ndim != 2 or current_factors.shape[0] != len(prior_std) or not np.isfinite(current_factors).all():
            raise ValueError("current factors must match the grouped prior")
    if not 1 <= rank <= len(prior_std) or parameter_step <= 0 or observation.sigma <= 0:
        raise ValueError("invalid sketch rank, parameter step or noise")
    rollout.set_action(trajectory.controller, trajectory.contact)
    run = parameterized_run(rollout, partition, observation, initial=initial)
    gains = []
    for phi in samples:
        sketch = randomized_sketch(run, phi, prior_std, observation.sigma, rank, parameter_step, seed)
        value = 0.5 * gain_against(sketch.U, current_factors)
        if not np.isfinite(value):
            raise ValueError("candidate information gain is not finite")
        gains.append(value)
    return float(np.dot(weights / weights.sum(), gains))


def plan_trajectory(
    controller: np.ndarray,
    contact: np.ndarray,
    score: Callable[[Trajectory], float],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    max_speed: float,
    dt: float,
    offset_limit: float,
    knots: int = 2,
    candidates: int = 8,
    seed: int = 0,
    gradient_steps: int = 0,
    action_step: float = 0.001,
    gradient_tolerance: float = 0.25,
) -> Plan:
    """Sample bounded candidates and optionally refine only stable gradients.

    The nominal zero-offset action is always evaluated. Sampled actions are
    uniform in the offset box and rejected if workspace or speed constraints
    are violated. A refinement is accepted only if feasible and better.
    Numerical gradients are compared at h and h/2 at every refinement step;
    an inconsistent or boundary gradient stops refinement, retaining the best
    sampled or previously refined trajectory.
    """
    if (not np.isfinite(offset_limit) or not np.isfinite(action_step)
        or not np.isfinite(gradient_tolerance) or offset_limit <= 0 or knots < 1
        or candidates < 1 or gradient_steps < 0 or action_step <= 0 or gradient_tolerance < 0):
        raise ValueError("invalid planner settings")
    rng = np.random.default_rng(seed)
    def evaluate(x):
        trajectory = make_trajectory(controller, contact, x.reshape(knots, 3))
        if not feasible(trajectory, lower, upper, max_speed, dt):
            return None
        value = float(score(trajectory))
        if not np.isfinite(value):
            raise ValueError("candidate score is not finite")
        return value

    zero = np.zeros((knots, 3))
    initial = evaluate(zero)
    if initial is None:
        raise ValueError("nominal trajectory violates workspace or speed bounds")
    best, best_score = zero.copy(), initial
    scores = [initial]
    for _ in range(candidates - 1):
        x = rng.uniform(-offset_limit, offset_limit, size=zero.shape)
        value = evaluate(x)
        if value is not None:
            scores.append(value)
            if value > best_score:
                best, best_score = x, value

    status = "not requested"
    for _ in range(gradient_steps):
        gradients = []
        for h in (action_step, action_step / 2):
            grad = np.zeros(best.size)
            for i in range(best.size):
                plus, minus = best.copy().reshape(-1), best.copy().reshape(-1)
                plus[i] += h
                minus[i] -= h
                if np.any(np.abs(plus) > offset_limit) or np.any(np.abs(minus) > offset_limit):
                    status = "gradient probe outside offset bounds"
                    break
                fp, fm = evaluate(plus), evaluate(minus)
                if fp is None or fm is None:
                    status = "gradient probe violates trajectory constraints"
                    break
                grad[i] = (fp - fm) / (2 * h)
            else:
                gradients.append(grad)
                continue
            break
        if len(gradients) != 2:
            break
        scale = max(np.linalg.norm(gradients[0]), np.linalg.norm(gradients[1]), 1e-12)
        if np.linalg.norm(gradients[0] - gradients[1]) / scale > gradient_tolerance:
            status = "unstable finite-difference gradient"
            break
        norm = np.linalg.norm(gradients[1])
        if norm < 1e-12:
            status = "zero gradient"
            break
        proposal = np.clip(best.reshape(-1) + action_step * gradients[1] / norm, -offset_limit, offset_limit).reshape(best.shape)
        value = evaluate(proposal)
        if value is None or value <= best_score:
            status = "no feasible improving gradient step"
            break
        best, best_score = proposal, value
        status = "improved with checked finite-difference gradient"
    return Plan(make_trajectory(controller, contact, best), best, best_score, np.asarray(scores), status)
