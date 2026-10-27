"""Spatial parameter groups and the phi <-> theta maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from asid.fim.case import Case
from asid.fim.simulator import SimConfig


def group_indices(groups: int) -> dict[str, slice]:
    """Parameter slices of ``[log k, log c, log m]`` for ``groups`` groups."""
    return {
        "k": slice(0, groups),
        "c": slice(groups, groups + 1),
        "m": slice(groups + 1, 2 * groups + 1),
    }


def kmeans(
    points: np.ndarray, groups: int, seed: int = 0, iterations: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if groups >= count:
        return np.arange(count), points.copy()
    rng = np.random.default_rng(seed)
    centers = [points[int(rng.integers(count))]]
    closest = np.linalg.norm(points - centers[0], axis=1) ** 2
    for _ in range(1, groups):
        index = int(rng.choice(count, p=closest / closest.sum()))
        centers.append(points[index])
        closest = np.minimum(closest, np.linalg.norm(points - points[index], axis=1) ** 2)
    centers = np.asarray(centers)

    labels = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        labels = cKDTree(centers).query(points)[1]
        counts = np.bincount(labels, minlength=groups)
        for empty in np.flatnonzero(counts == 0):
            distance = np.linalg.norm(points - centers[labels], axis=1)
            index = int(np.argmax(distance))
            labels[index] = empty
            counts = np.bincount(labels, minlength=groups)
        updated = np.stack(
            [points[labels == group].mean(axis=0) for group in range(groups)]
        )
        if np.allclose(updated, centers, atol=1e-10, rtol=0.0):
            break
        centers = updated
    return labels, centers


@dataclass(frozen=True)
class Partition:
    node_centroids: np.ndarray
    node_groups: np.ndarray
    spring_groups: np.ndarray
    num_object_springs: int
    num_springs: int
    phi0: np.ndarray
    nominal_spring_Y: float

    @property
    def groups(self) -> int:
        return len(self.node_centroids)

    @property
    def dimension(self) -> int:
        return 2 * self.groups + 1

    @property
    def indices(self) -> dict[str, slice]:
        return group_indices(self.groups)

    def spring_Y(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=np.float64)
        values = np.full(self.num_springs, self.nominal_spring_Y)
        values[: self.num_object_springs] = np.exp(phi[self.indices["k"]])[
            self.spring_groups
        ]
        return values

    def masses(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=np.float64)
        return np.exp(phi[self.indices["m"]])[self.node_groups]

    def dashpot(self, phi: np.ndarray) -> float:
        return float(np.exp(np.asarray(phi)[self.indices["c"]][0]))

    def apply(self, rollout, phi: np.ndarray) -> None:
        rollout.set_spring_Y(self.spring_Y(phi))
        rollout.set_masses(self.masses(phi))
        rollout.set_dashpot(self.dashpot(phi))


def build_partition(
    case: Case,
    groups: int,
    seed: int = 0,
    config: SimConfig = SimConfig(),
) -> Partition:
    node_groups, node_centroids = kmeans(case.nodes, groups, seed=seed)
    spring_groups = cKDTree(node_centroids).query(case.object_midpoints)[1]

    phi0 = np.concatenate(
        [
            np.full(groups, np.log(config.spring_Y)),
            [np.log(config.dashpot_damping)],
            np.full(groups, np.log(config.total_mass / len(case.nodes))),
        ]
    )
    return Partition(
        node_centroids=node_centroids,
        node_groups=node_groups,
        spring_groups=spring_groups,
        num_object_springs=case.num_object_springs,
        num_springs=len(case.springs),
        phi0=phi0,
        nominal_spring_Y=config.spring_Y,
    )
