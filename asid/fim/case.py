"""Load a PhysTwin case and build the reduced analysis mesh.

The mesh is a seeded farthest-point subsample of the case's structure points.
Object springs are a radius graph; the sparse controller points are attached by
the same tether rule the trainer uses, which the simulator requires before it
will update the dense contact trajectory.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from qqtt.utils.controller_collider import hollow_controller_points


@dataclass(frozen=True)
class Case:
    name: str
    nodes: np.ndarray
    springs: np.ndarray
    rest_lengths: np.ndarray
    num_object_springs: int
    controller: np.ndarray
    contact: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.controller)

    @property
    def object_midpoints(self) -> np.ndarray:
        return object_midpoints(self.nodes, self.springs, self.num_object_springs)


def object_midpoints(nodes: np.ndarray, springs: np.ndarray, count: int) -> np.ndarray:
    """Midpoints of the first ``count`` springs, i.e. the object springs."""
    pairs = np.asarray(springs, dtype=np.int64)[: int(count)]
    points = np.asarray(nodes, dtype=np.float64)
    return 0.5 * (points[pairs[:, 0]] + points[pairs[:, 1]])


def farthest_point_indices(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if count >= len(points):
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(points)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    distance = np.linalg.norm(points - points[first], axis=1)
    for index in range(1, count):
        nxt = int(np.argmax(distance))
        selected[index] = nxt
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return selected


def build_object_springs(
    points: np.ndarray,
    radius: float,
    max_neighbours: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    tree = cKDTree(points)
    neighbours = min(int(max_neighbours), len(points))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, point in enumerate(points):
        distances, indices = tree.query(
            point, k=neighbours, distance_upper_bound=float(radius)
        )
        for distance, j in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if not np.isfinite(distance) or j >= len(points) or j == i:
                continue
            if distance <= 1e-4:
                continue
            pair = (i, int(j)) if i < int(j) else (int(j), i)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    springs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    rest_lengths = np.linalg.norm(
        points[springs[:, 0]] - points[springs[:, 1]], axis=1
    ).astype(np.float32)
    return springs, rest_lengths


def build_controller_tethers(
    nodes: np.ndarray,
    controller: np.ndarray,
    radius: float,
    max_neighbours: int,
    num_object_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(nodes)
    neighbours = min(int(max_neighbours), len(nodes))
    springs: list[tuple[int, int]] = []
    rest_lengths: list[float] = []
    for controller_index, point in enumerate(controller):
        distances, indices = tree.query(
            point, k=neighbours, distance_upper_bound=float(radius)
        )
        for distance, j in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if not np.isfinite(distance) or distance <= 1e-4 or j >= len(nodes):
                continue
            springs.append((num_object_nodes + controller_index, int(j)))
            rest_lengths.append(float(distance))
    return (
        np.asarray(springs, dtype=np.int32).reshape(-1, 2),
        np.asarray(rest_lengths, dtype=np.float32),
    )


def load_case(
    case_dir: str | Path,
    num_nodes: int = 500,
    *,
    node_seed: int = 0,
    object_radius: float = 0.04,
    object_max_neighbours: int = 30,
    controller_radius: float = 0.04,
    controller_max_neighbours: int = 50,
    contact_voxel_size: float = 0.003,
    max_frames: int = 120,
) -> Case:
    case_dir = Path(case_dir)
    with (case_dir / "final_data.pkl").open("rb") as file:
        data = pickle.load(file)

    points = np.concatenate(
        [data["object_points"][0], data["surface_points"], data["interior_points"]],
        axis=0,
    )
    nodes = points[farthest_point_indices(points, num_nodes, node_seed)]

    object_springs, object_rest = build_object_springs(
        nodes, object_radius, object_max_neighbours
    )
    controller = np.asarray(data["controller_points"][:max_frames], dtype=np.float64)
    contact = hollow_controller_points(
        np.asarray(data["controller_points_dense"][:max_frames], dtype=np.float64),
        contact_voxel_size,
    )
    tethers, tether_rest = build_controller_tethers(
        nodes,
        controller[0],
        controller_radius,
        controller_max_neighbours,
        len(nodes),
    )
    if len(controller) != len(contact):
        raise ValueError("controller and dense contact trajectories disagree in length")

    return Case(
        name=case_dir.name,
        nodes=nodes.astype(np.float32),
        springs=np.concatenate([object_springs, tethers], axis=0),
        rest_lengths=np.concatenate([object_rest, tether_rest], axis=0),
        num_object_springs=len(object_springs),
        controller=controller.astype(np.float32),
        contact=contact.astype(np.float32),
    )
