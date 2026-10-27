"""Myopic sequential design under the accumulated posterior.

After a set of actions the posterior precision in whitened coordinates is
``P = I + sum H_a`` with ``H_a`` the whitened Fisher of each action. The next
action is the one with the largest ``log det(I + P^{-1} H)``, i.e. the one that
adds the most information about what is still unknown.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def precision(information: Sequence[np.ndarray], dimension: int) -> np.ndarray:
    """``I + sum H``; starts from the prior precision."""
    total = np.eye(int(dimension))
    for matrix in information:
        total = total + np.asarray(matrix, dtype=np.float64)
    return total


def marginal_gain(precision_matrix: np.ndarray, information: np.ndarray) -> float:
    """``log det(I + P^{-1} H)`` by two Cholesky-safe determinants."""
    matrix = np.asarray(precision_matrix, dtype=np.float64)
    added = matrix + np.asarray(information, dtype=np.float64)
    return float(np.linalg.slogdet(added)[1] - np.linalg.slogdet(matrix)[1])


def select(
    precision_matrix: np.ndarray, information: Mapping[str, np.ndarray]
) -> tuple[str, dict[str, float]]:
    """Name with the largest posterior-conditioned gain, plus every score."""
    scores = {
        name: marginal_gain(precision_matrix, matrix)
        for name, matrix in information.items()
    }
    best = max(scores, key=lambda name: scores[name])
    return best, scores
