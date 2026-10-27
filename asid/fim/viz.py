"""Partition and Fisher overlay figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _equal_axes(ax, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    span = float((high - low).max())
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _limits(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2, 98])
    if low == high:
        return float(low - 0.5), float(high + 0.5)
    return float(low), float(high)


def point_information(
    jac: np.ndarray, prior_std: np.ndarray, sigma: float, nodes: int
) -> np.ndarray:
    """Standalone local Gaussian information gain from each observation node.

    ``jac`` has parameter rows and flattened (frame, node, xyz) columns. Each
    value uses all selected observations of one node with the original prior;
    the values do not add up to the joint gain because nodes can be redundant.
    """
    jac = np.asarray(jac, dtype=np.float64)
    prior_std = np.asarray(prior_std, dtype=np.float64)
    if sigma <= 0 or nodes <= 0 or jac.ndim != 2:
        raise ValueError("sigma and nodes must be positive and jac must be 2D")
    if jac.shape[0] != len(prior_std) or jac.shape[1] % (nodes * 3):
        raise ValueError("jac shape must match parameters and whole 3D node frames")
    weighted = (jac * prior_std[:, None] / sigma).reshape(
        len(prior_std), -1, nodes, 3
    )
    values = np.empty(nodes, dtype=np.float64)
    for node in range(nodes):
        columns = weighted[:, :, node, :].reshape(len(prior_std), -1)
        gram = columns @ columns.T if columns.shape[0] <= columns.shape[1] else columns.T @ columns
        chol = np.linalg.cholesky(np.eye(len(gram)) + gram)
        values[node] = np.log(np.diag(chol)).sum()
    return values


def plot_fim_matrix(matrix: np.ndarray, path: str | Path) -> Path:
    """Plot the grouped Fisher matrix in log-parameter coordinates."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Fisher matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("Fisher matrix must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    limit = max(float(np.abs(matrix).max()), 1e-30)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set(xlabel="grouped parameter", ylabel="grouped parameter", title="Grouped Fisher matrix")
    fig.colorbar(image, ax=ax, label="F")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_point_information(nodes: np.ndarray, values: np.ndarray, path: str | Path) -> Path:
    """Color observed nodes by standalone local Gaussian information gain."""
    nodes = np.asarray(nodes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or values.shape != (len(nodes),):
        raise ValueError("expected nodes (N, 3) and values (N,)")
    if not np.isfinite(nodes).all() or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("node positions and nonnegative information must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.log10(np.maximum(values, 1e-30))
    scatter = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=colors, cmap="viridis", s=12)
    _equal_axes(ax, nodes)
    ax.set_title("Standalone per-node local information gain")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="log10 gain (nats)")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_fim_overlay(
    nodes: np.ndarray,
    midpoints: np.ndarray,
    node_groups: np.ndarray,
    spring_groups: np.ndarray,
    diag: np.ndarray,
    indices: dict[str, slice],
    path: str | Path,
    title: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = int(node_groups.max()) + 1
    points = np.concatenate([nodes, midpoints], axis=0)

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(
        nodes[:, 0], nodes[:, 1], nodes[:, 2],
        c=node_groups, cmap="turbo", vmin=0, vmax=max(groups - 1, 1), s=8,
    )
    ax.scatter(
        midpoints[:, 0], midpoints[:, 1], midpoints[:, 2],
        c="black", s=1, alpha=0.35,
    )
    ax.set_title(f"spatial partition (g={groups})")
    _equal_axes(ax, points)

    for column, key, label, sample, sample_groups in (
        (2, "k", "log10 F_kk", midpoints, spring_groups),
        (3, "m", "log10 F_mm", nodes, node_groups),
    ):
        values = np.log10(np.maximum(diag[indices[key]], 1e-30))
        low, high = _limits(values[sample_groups])
        ax = fig.add_subplot(1, 3, column, projection="3d")
        gray = np.full(len(nodes), 0.75)
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=gray, s=4, alpha=0.5)
        scatter = ax.scatter(
            sample[:, 0], sample[:, 1], sample[:, 2],
            c=values[sample_groups],
            cmap="viridis",
            vmin=low,
            vmax=high,
            s=10,
        )
        ax.set_title(label)
        _equal_axes(ax, points)
        fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)

    dashpot = float(np.log10(max(diag[indices["c"]][0], 1e-30)))
    fig.suptitle(f"{title}  |  log10 F_cc = {dashpot:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
