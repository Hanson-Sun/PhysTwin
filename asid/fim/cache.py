"""On-disk caches for sketch factors and exact Fisher scores.

A factor is not cacheable across trajectories: it is local to a trajectory, a
body state and the parameter point. What is cacheable is the factor itself,
keyed by exactly the inputs that made it, and reused only when the same
trajectory is measured again.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from asid.fim.sketch import Sketch

# Bump when the simulator, sketch or observation model changes: entries keyed by
# an older version are then ignored instead of silently reused.
CACHE_VERSION = 1


def digest(*parts) -> str:
    hasher = hashlib.sha1()
    for part in parts:
        if isinstance(part, np.ndarray):
            hasher.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
        else:
            hasher.update(repr(part).encode())
        hasher.update(b"|")
    return hasher.hexdigest()


def _trajectory_key(case, config, phi0, controller, contact, step, sigma) -> str:
    return digest(
        CACHE_VERSION,
        case.name,
        len(case.nodes),
        dataclasses.asdict(config),
        np.asarray(phi0),
        np.asarray(controller),
        np.asarray(contact),
        step,
        sigma,
    )


def factor_key(case, config, phi0, controller, contact, rank, step, sigma, seed) -> str:
    """Everything a ``randomized_sketch`` result depends on, plus a version."""
    return digest(
        _trajectory_key(case, config, phi0, controller, contact, step, sigma), rank, seed
    )


def exact_key(case, config, phi0, controller, contact, step, sigma) -> str:
    """Everything an exact Fisher log-det depends on. No rank or seed: exact is exact."""
    return digest(
        "exact", _trajectory_key(case, config, phi0, controller, contact, step, sigma)
    )


class _Store:
    """A directory of ``<key>.npz`` entries with hit and miss accounting."""

    label = "cache"

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else None
        self.hits = 0
        self.misses = 0

    def _read(self, path: Path):
        raise NotImplementedError

    def get(self, key: str):
        if self.root is None:
            self.misses += 1
            return None
        path = self.root / f"{key}.npz"
        if not path.is_file():
            self.misses += 1
            return None
        self.hits += 1
        return self._read(path)

    def _path(self, key: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{key}.npz"

    def report(self) -> str:
        total = self.hits + self.misses
        return f"{self.label}: {self.hits}/{total} hits"


class FactorCache(_Store):
    """Sketch factors on disk, keyed by :func:`factor_key`."""

    label = "factor cache"

    def _read(self, path: Path) -> Sketch:
        data = np.load(path)
        return Sketch(
            directions=data["directions"],
            JQ=data["JQ"],
            T=data["T"],
            U=data["U"],
            gain=float(data["gain"]),
        )

    def put(self, key: str, sketch: Sketch) -> None:
        if self.root is None:
            return
        np.savez(
            self._path(key),
            directions=sketch.directions,
            JQ=sketch.JQ,
            T=sketch.T,
            U=sketch.U,
            gain=sketch.gain,
        )


class ExactCache(_Store):
    """Exact Fisher log-dets on disk, keyed by :func:`exact_key`.

    The exact metric costs ``2P`` rollouts per trajectory, so a shortlist is
    worth caching; unlike a sketch factor it does not depend on rank or seed.
    """

    label = "exact cache"

    def _read(self, path: Path):
        data = np.load(path)
        return float(data["gain"]), data["matrix"]

    def put(self, key: str, gain: float, matrix: np.ndarray) -> None:
        if self.root is None:
            return
        np.savez(self._path(key), gain=gain, matrix=matrix)
