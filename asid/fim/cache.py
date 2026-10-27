"""Caches for work that does not depend on the trajectory under test.

A factor is not cacheable across trajectories: it is local to an action, a body
state and the parameter point, so every candidate is scored with its own
rollouts. What is cacheable is everything around that -- one simulator instance
reused across actions, and factors keyed by exactly the inputs that made them.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from asid.fim.sketch import Sketch

# Bump when the simulator, sketch or observation model changes: factors keyed by
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


def factor_key(case, config, phi0, controller, contact, rank, step, sigma, seed) -> str:
    """Everything a ``randomized_sketch`` result depends on, plus a version."""
    return digest(
        CACHE_VERSION,
        case.name,
        len(case.nodes),
        dataclasses.asdict(config),
        np.asarray(phi0),
        np.asarray(controller),
        np.asarray(contact),
        rank,
        step,
        sigma,
        seed,
    )


class ActionRunner:
    """One simulator instance, re-armed for each trajectory.

    ``Rollout`` compiles Warp kernels and allocates its graph, so building one
    per candidate costs seconds each. ``set_action`` swaps the controller and
    contact trajectories and ``run`` restarts from the case state, so a single
    instance is enough as long as evaluation stays sequential.
    """

    def __init__(self, case, config):
        from asid.fim.simulator import Rollout  # keeps this module Warp-free

        self.rollout = Rollout(case, config)

    def run(self, controller, contact, partition, observation, initial=None):
        """Return ``phi -> observation vector`` for one trajectory."""
        def evaluate(phi):
            self.rollout.set_action(controller, contact)
            partition.apply(self.rollout, phi)
            return observation.vector(self.rollout.run(initial=initial))
        return evaluate


class FactorCache:
    """Sketch factors on disk, keyed by :func:`factor_key`."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else None
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Sketch | None:
        if self.root is None:
            self.misses += 1
            return None
        path = self.root / f"{key}.npz"
        if not path.is_file():
            self.misses += 1
            return None
        data = np.load(path)
        self.hits += 1
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
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.root / f"{key}.npz",
            directions=sketch.directions,
            JQ=sketch.JQ,
            T=sketch.T,
            U=sketch.U,
            gain=sketch.gain,
        )

    def report(self) -> str:
        total = self.hits + self.misses
        return f"factor cache: {self.hits}/{total} hits"
