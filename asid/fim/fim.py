"""Exact Fisher information in grouped parameter coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prior:
    spring_log_std: float = 0.5
    dashpot_log_std: float = 0.5
    mass_log_std: float = 0.3

    def std(self, groups: int) -> np.ndarray:
        return np.concatenate(
            [
                np.full(groups, self.spring_log_std),
                [self.dashpot_log_std],
                np.full(groups, self.mass_log_std),
            ]
        )


def posterior_std(prior_std: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Marginal posterior std given the precision ``I + H`` in whitened space."""
    covariance = np.linalg.inv(np.asarray(precision, dtype=np.float64))
    return np.asarray(prior_std, dtype=np.float64) * np.sqrt(
        np.maximum(np.diag(covariance), 0.0)
    )


@dataclass(frozen=True)
class Fisher:
    matrix: np.ndarray
    prior_std: np.ndarray

    @classmethod
    def from_jacobian(
        cls, jac: np.ndarray, sigma: float, prior_std: np.ndarray
    ) -> "Fisher":
        return cls(matrix=(jac / sigma) @ (jac / sigma).T, prior_std=prior_std)

    @property
    def whitened(self) -> np.ndarray:
        return self.matrix * self.prior_std[:, None] * self.prior_std[None, :]

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.whitened)

    def logdet(self) -> float:
        return float(np.linalg.slogdet(np.eye(len(self.matrix)) + self.whitened)[1])

    def posterior_std(self) -> np.ndarray:
        return posterior_std(self.prior_std, np.eye(len(self.matrix)) + self.whitened)
