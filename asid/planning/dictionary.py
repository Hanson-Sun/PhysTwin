"""Sketch factors for a set of candidate actions.

``shared`` draws one subspace for every action, so standalone gains are compared
with common random numbers. ``independent`` draws one per action, which keeps the
union from being capped at ``rank`` dimensions when factors are accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from asid.fim.sketch import randomized_sketch


@dataclass(frozen=True)
class Entry:
    """One action's rank-``rank`` factor ``U`` of its whitened Fisher."""

    name: str
    gain: float
    U: np.ndarray
    JQ: np.ndarray | None = None


SUBSPACES = ("shared", "independent")


def build_dictionary(
    requests: Sequence[tuple[str, Callable[[np.ndarray], np.ndarray]]],
    phi0: np.ndarray,
    prior_std: np.ndarray,
    sigma: float,
    rank: int,
    step: float,
    seed: int = 0,
    subspace: str = "shared",
    progress: Callable[[int, int], None] | None = None,
) -> list[Entry]:
    """Sketch every ``(name, run)`` request at the same ``phi0``."""
    if subspace not in SUBSPACES:
        raise ValueError(f"subspace must be one of {SUBSPACES}: {subspace!r}")
    entries = []
    for index, (name, run) in enumerate(requests):
        sketch = randomized_sketch(
            run,
            phi0,
            prior_std,
            sigma,
            rank=rank,
            step=step,
            seed=seed if subspace == "shared" else seed + 1 + index,
        )
        if not (np.isfinite(sketch.JQ).all() and np.isfinite(sketch.gain)):
            raise RuntimeError(f"non-finite sketch for action {name!r}")
        entries.append(Entry(name, sketch.gain, sketch.U, sketch.JQ))
        if progress is not None:
            progress(index + 1, len(requests))
    return entries
