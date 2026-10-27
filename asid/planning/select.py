"""Greedy selection over sketch factors; no rollouts, no full Fisher."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from asid.fim.sketch import gain_against
from asid.planning.dictionary import Entry


def accumulate(current: np.ndarray | None, U: np.ndarray) -> np.ndarray:
    return U if current is None else np.concatenate([current, U], axis=1)


def subspace_entry(name: str, whitened: np.ndarray, Q: np.ndarray) -> Entry:
    """Exact factor of ``whitened`` restricted to the subspace ``Q``.

    Isolates the rank cap of a shared sketch subspace from derivative noise.
    """
    T = Q.T @ np.asarray(whitened, dtype=np.float64) @ Q
    values, vectors = np.linalg.eigh(0.5 * (T + T.T))
    values = np.maximum(values, 0.0)
    U = np.asarray(Q, dtype=np.float64) @ (vectors * np.sqrt(values))
    gain = float(np.linalg.slogdet(np.eye(len(values)) + np.diag(values))[1])
    return Entry(name=name, gain=gain, U=U)


def greedy(entries: Sequence[Entry], count: int) -> list[tuple[Entry, float]]:
    """Pick the largest marginal gain each round; returns ``(entry, gain)``."""
    remaining = list(entries)
    chosen: list[tuple[Entry, float]] = []
    factors = None
    while remaining and len(chosen) < count:
        gain, best = max(
            ((gain_against(entry.U, factors), entry) for entry in remaining),
            key=lambda item: item[0],
        )
        chosen.append((best, gain))
        factors = accumulate(factors, best.U)
        remaining.remove(best)
    return chosen
