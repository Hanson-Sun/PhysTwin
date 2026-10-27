"""Finite-difference Jacobian operators in grouped parameter space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Observation:
    """Selected node positions with scalar noise std in the returned coordinates.

    For ``reduce='mean'``, independent per-frame noise with standard deviation
    ``sigma_frame`` has output standard deviation ``sigma_frame / sqrt(n)``.
    Correlated noise is not represented by this scalar-noise observation model.
    """

    frames: np.ndarray
    sigma: float = 0.005
    reduce: str = "none"

    def vector(self, trajectory) -> np.ndarray:
        values = trajectory[self.frames].detach().cpu().numpy().astype(np.float64)
        if self.reduce == "mean":
            values = values.mean(axis=0)
        elif self.reduce != "none":
            raise ValueError(f"unknown reduction: {self.reduce!r}")
        return values.reshape(-1)


def default_observation(
    frames: int, stride: int = 2, sigma: float = 0.005, reduce: str = "none"
) -> Observation:
    return Observation(
        frames=np.arange(0, frames, stride), sigma=sigma, reduce=reduce
    )

def parameterized_run(rollout, partition, observation: Observation, initial=None):
    """Map ``phi`` to the observation, optionally starting from a deformed state."""

    def run(phi: np.ndarray) -> np.ndarray:
        partition.apply(rollout, phi)
        return observation.vector(rollout.run(initial=initial))

    return run


def jacobian(
    run: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    step: float,
    indices: Sequence[int] | None = None,
    central: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Rows are ``d y / d phi_p``; one-sided reuses the nominal rollout."""
    phi0 = np.asarray(phi0, dtype=np.float64)
    if indices is None:
        indices = range(len(phi0))
    base = None if central else run(phi0)
    indices = list(indices)
    rows = []
    for done, index in enumerate(indices, start=1):
        delta = np.zeros_like(phi0)
        delta[index] = step
        plus = run(phi0 + delta)
        if central:
            rows.append((plus - run(phi0 - delta)) / (2.0 * step))
        else:
            rows.append((plus - base) / step)
        if progress is not None:
            progress(done, len(indices))
    return np.stack(rows)


def directional_derivatives(
    run: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    directions: np.ndarray,
    step: float,
) -> np.ndarray:
    """Apply ``J`` to each column: returns ``(D, ell)`` by central differences."""
    phi0 = np.asarray(phi0, dtype=np.float64)
    columns = []
    for column in range(directions.shape[1]):
        delta = step * directions[:, column]
        columns.append((run(phi0 + delta) - run(phi0 - delta)) / (2.0 * step))
    return np.stack(columns, axis=1)
