"""Randomized q = 0 estimator of the whitened Fisher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from asid.fim.operators import directional_derivatives


@dataclass(frozen=True)
class Sketch:
    directions: np.ndarray
    JQ: np.ndarray
    T: np.ndarray
    U: np.ndarray
    gain: float


def random_subspace(dimension: int, rank: int, seed: int = 0) -> np.ndarray:
    """Orthonormal basis of a random Rademacher sketch subspace."""
    rng = np.random.default_rng(seed)
    if not 1 <= rank <= dimension:
        raise ValueError("rank must be between 1 and dimension")
    # Reject rank-deficient draws: QR would otherwise add arbitrary directions.
    for _ in range(100):
        omega = rng.choice([-1.0, 1.0], size=(int(dimension), int(rank)))
        if np.linalg.matrix_rank(omega) == rank:
            return np.linalg.qr(omega, mode="reduced")[0]
    raise RuntimeError("could not draw a full-rank random subspace")


def randomized_sketch(
    run: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    prior_std: np.ndarray,
    sigma: float,
    rank: int,
    step: float,
    seed: int = 0,
) -> Sketch:
    phi0 = np.asarray(phi0, dtype=np.float64)
    rank = min(int(rank), len(phi0))
    Q = random_subspace(len(phi0), rank, seed)

    directions = prior_std[:, None] * Q
    JQ = directional_derivatives(run, phi0, directions, step) / sigma
    T = JQ.T @ JQ
    values, vectors = np.linalg.eigh(T)
    values = np.maximum(values, 0.0)
    U = (Q @ vectors) * np.sqrt(values)
    gain = float(np.linalg.slogdet(np.eye(rank) + T)[1])
    return Sketch(directions=directions, JQ=JQ, T=T, U=U, gain=gain)


def gain_against(U: np.ndarray, current: np.ndarray | None) -> float:
    """log det(I + U^T H_cur^{-1} U) with H_cur = I + sum of prior factors."""
    if current is None or current.shape[1] == 0:
        return float(np.linalg.slogdet(np.eye(U.shape[1]) + U.T @ U)[1])
    small = np.eye(current.shape[1]) + current.T @ current
    solved = np.linalg.solve(small, current.T @ U)
    correction = np.eye(U.shape[1]) + U.T @ U - U.T @ current @ solved
    correction = 0.5 * (correction + correction.T)
    chol = np.linalg.cholesky(correction)
    return float(2.0 * np.log(np.diag(chol)).sum())
