"""Sketch factors for a set of candidate actions.

``shared`` draws one subspace for every action, so standalone gains are compared
with common random numbers. ``independent`` draws one per action, which keeps the
union from being capped at ``rank`` dimensions when factors are accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from asid.fim.sketch import randomized_sketch

if TYPE_CHECKING:  # pragma: no cover - keeps the module free of the Warp import
    from asid.fim.cache import FactorCache


@dataclass(frozen=True)
class Entry:
    """One action's rank-``rank`` factor ``U`` of its whitened Fisher."""

    name: str
    gain: float
    U: np.ndarray
    JQ: np.ndarray | None = None


SUBSPACES = ("shared", "independent")


def subspace_seed(seed: int, index: int, subspace: str) -> int:
    """Seed a sketch: common random numbers, or one subspace per action."""
    return int(seed) if subspace == "shared" else int(seed) + 1 + index


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
    cache: "FactorCache | None" = None,
    key: Callable[[int, str], str] | None = None,
) -> list[Entry]:
    """Sketch every ``(name, run)`` request at the same ``phi0``.

    ``cache`` and ``key`` are optional: pass a :class:`~asid.fim.cache.FactorCache`
    and a key builder to reuse factors whose action, parameters and settings are
    unchanged. A factor is never shared between different actions, because the
    key includes the trajectory arrays.
    """
    if subspace not in SUBSPACES:
        raise ValueError(f"subspace must be one of {SUBSPACES}: {subspace!r}")
    entries = []
    for index, (name, run) in enumerate(requests):
        cache_key = key(index, name) if cache is not None and key is not None else None
        sketch = cache.get(cache_key) if cache_key is not None else None
        if sketch is None:
            sketch = randomized_sketch(
                run,
                phi0,
                prior_std,
                sigma,
                rank=rank,
                step=step,
                seed=subspace_seed(seed, index, subspace),
            )
            if cache_key is not None:
                cache.put(cache_key, sketch)
        if not (np.isfinite(sketch.JQ).all() and np.isfinite(sketch.gain)):
            raise RuntimeError(f"non-finite sketch for action {name!r}")
        entries.append(Entry(name, sketch.gain, sketch.U, sketch.JQ))
        if progress is not None:
            progress(index + 1, len(requests))
    return entries
