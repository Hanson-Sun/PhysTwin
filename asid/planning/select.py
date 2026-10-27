"""Greedy selection over sketch factors; no rollouts, no full Fisher."""

from __future__ import annotations

from typing import Mapping, Sequence

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


def shortlist(entries: Sequence[Entry], count: int) -> list[Entry]:
    """The ``count`` best entries by sketch gain, keeping name order on ties."""
    if count < 1:
        raise ValueError("a shortlist needs at least one entry")
    return sorted(entries, key=lambda entry: entry.gain, reverse=True)[:count]


def exact_ordered(
    entries: Sequence[Entry], gains: Mapping[str, float]
) -> list[Entry]:
    """Order a shortlist by exact log-det.

    The sketch ranks the family cheaply but its finite-difference protocol
    differs from the exact one, so at g = 20 it can invert close candidates
    (measured: az270 233.23 > az090 230.75 by sketch against 628.37 < 692.67
    exactly). The decision therefore comes from the exact metric over the
    shortlist; every shortlisted entry must have a gain.
    """
    missing = [entry.name for entry in entries if entry.name not in gains]
    if missing:
        raise ValueError(f"exact gains missing for shortlisted actions: {missing}")
    return sorted(entries, key=lambda entry: gains[entry.name], reverse=True)


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
