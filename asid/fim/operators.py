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

    def batch(self, trajectory, instances: int) -> np.ndarray:
        """Per-instance observation vectors of a ``(T, k*N, 3)`` trajectory."""
        instances = int(instances)
        if instances < 1:
            raise ValueError("instances must be positive")
        values = trajectory[self.frames].detach().cpu().numpy().astype(np.float64)
        values = values.reshape(len(self.frames), instances, -1, 3)
        if self.reduce == "mean":
            values = values.mean(axis=0)
        elif self.reduce != "none":
            raise ValueError(f"unknown reduction: {self.reduce!r}")
        else:
            values = values.transpose(1, 0, 2, 3)
        return values.reshape(instances, -1)


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


def _run_in_chunks(
    run_batch: Callable[[np.ndarray], np.ndarray],
    phi_batch: np.ndarray,
    chunk: int | None,
) -> np.ndarray:
    """Run a ``phi_batch -> rows`` map in chunks and concatenate the rows."""
    chunk = max(1, min(int(chunk or len(phi_batch)), len(phi_batch)))
    rows = None
    for start in range(0, len(phi_batch), chunk):
        block = np.asarray(run_batch(phi_batch[start : start + chunk]))
        if block.ndim != 2:
            raise ValueError("batched runs must return a 2-D observation array")
        if rows is None:
            rows = np.empty((len(phi_batch), block.shape[1]), dtype=block.dtype)
        rows[start : start + len(block)] = block
    return rows


def jacobian_batched(
    run_batch: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    step: float,
    indices: Sequence[int] | None = None,
    chunk: int | None = None,
) -> np.ndarray:
    """Central differences of a batched ``phi_batch -> observations`` map.

    The batch is the ``+step`` rows followed by the ``-step`` rows, so row ``i``
    of the result is ``(Y[i] - Y[i + n]) / (2 * step)``. Rows are stored before
    they are paired, so a chunk may split the two halves.
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    dimension = len(phi0)
    if indices is None:
        indices = range(dimension)
    indices = list(indices)
    count = len(indices)
    if count == 0:
        return np.empty((0, 0), dtype=np.float64)
    directions = np.eye(dimension)[indices]
    batch = np.concatenate([phi0 + step * directions, phi0 - step * directions])
    if chunk is None:
        # At most 2g instances: half the exact batch, which is the memory knob
        # the plan documents. P = 2g + 1.
        chunk = max(1, 2 * ((dimension - 1) // 2))
    values = _run_in_chunks(run_batch, batch, chunk)
    return (values[:count] - values[count:]) / (2.0 * step)


def directional_derivatives_batched(
    run_batch: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    directions: np.ndarray,
    step: float,
    chunk: int | None = None,
) -> np.ndarray:
    """Apply ``J`` to each column with a batched ``phi_batch -> observations``."""
    phi0 = np.asarray(phi0, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    count = directions.shape[1]
    offsets = step * directions.T
    batch = np.concatenate([phi0 + offsets, phi0 - offsets])
    values = _run_in_chunks(run_batch, batch, chunk)
    return (values[:count] - values[count:]).T / (2.0 * step)
