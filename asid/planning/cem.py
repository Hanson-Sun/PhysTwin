"""Cross-entropy method for bounded continuous control vectors.

Elite refitting replaces the argmax of the uniform sampler: the distribution
narrows around the best samples instead of throwing darts at a fixed box, and
elite ranking is scale-free, so it does not care that the information score is a
log-determinant rather than a cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CemResult:
    vector: np.ndarray
    score: float
    history: list[float]
    evaluations: int


def cem(
    evaluate: Callable[[np.ndarray], float | None],
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    samples: int = 8,
    iterations: int = 3,
    elites: int = 2,
    init_std: float | None = None,
    smoothing: float = 0.7,
    min_std: float = 1e-4,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
    seed: int = 0,
) -> CemResult:
    """Maximize ``evaluate`` with an elite-refitted diagonal Gaussian.

    ``evaluate`` returns ``None`` for an infeasible vector; those samples are
    dropped instead of being penalised, so a constraint never masquerades as a
    low score. ``project`` maps a sample back onto the feasible set (ordering,
    for a time warp). The initial vector is always evaluated and always competes
    for the elites, so the result never scores below it.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("bounds must be two matching 1-D arrays")
    # ``lower == upper`` pins a coordinate (e.g. a trajectory's start pose);
    # only ``lower > upper`` is a broken box.
    if np.any(lower > upper) or samples < 1 or iterations < 1 or elites < 1:
        raise ValueError("invalid CEM settings")
    if not 0.0 <= smoothing < 1.0 or min_std <= 0.0:
        raise ValueError("invalid smoothing or std floor")

    def clip(vector: np.ndarray) -> np.ndarray:
        vector = np.clip(vector, lower, upper)
        return project(vector) if project is not None else vector

    mean = clip(np.asarray(initial, dtype=np.float64).copy())
    score = evaluate(mean)
    if score is None:
        raise ValueError("initial vector is infeasible")
    free = upper > lower
    std = np.where(
        free,
        (upper - lower)[free].mean() * 0.5 if init_std is None else init_std,
        0.0,
    )

    best, best_score = mean.copy(), float(score)
    history = [best_score]
    evaluations = 1
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        scored = [(best_score, best)]
        for _ in range(samples):
            candidate = clip(mean + std * rng.standard_normal(mean.size))
            value = evaluate(candidate)
            evaluations += 1
            if value is not None and np.isfinite(value):
                scored.append((float(value), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored[0][0] > best_score:
            best, best_score = scored[0][1].copy(), scored[0][0]
        chosen = np.stack([vector for _, vector in scored[: max(elites, 1)]])
        mean = chosen.mean(axis=0)
        spread = chosen.std(axis=0) if len(chosen) > 1 else std
        std = np.where(free, np.maximum(smoothing * std + (1.0 - smoothing) * spread, min_std), 0.0)
        history.append(best_score)
    return CemResult(best, best_score, history, evaluations)
