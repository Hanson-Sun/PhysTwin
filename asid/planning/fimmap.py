"""Top-view heatmaps of the Fisher diagonal, one column per action.

Rows separate the stiffness and mass blocks so the same scale is used across
actions and the change in distribution is visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

ROWS = (
    ("k", "midpoints", r"$\log_{10} F_{kk}$  (springs)"),
    ("m", "nodes", r"$\log_{10} F_{mm}$  (nodes)"),
)


def fisher_diagonal(matrix: np.ndarray, groups: int) -> dict[str, np.ndarray]:
    """Split ``diag(F)`` into stiffness, dashpot and mass blocks."""
    diag = np.diag(np.asarray(matrix, dtype=np.float64))
    return {
        "k": diag[:groups],
        "c": diag[groups : groups + 1],
        "m": diag[groups + 1 :],
    }


def group_values(
    diagonal: Mapping[str, np.ndarray],
    spring_groups: np.ndarray,
    node_groups: np.ndarray,
) -> dict[str, np.ndarray]:
    """Expand per-group information onto springs and nodes."""
    return {
        "k": np.asarray(diagonal["k"])[np.asarray(spring_groups, dtype=np.int64)],
        "c": np.asarray(diagonal["c"]),
        "m": np.asarray(diagonal["m"])[np.asarray(node_groups, dtype=np.int64)],
    }


def square_extent(points: np.ndarray, pad: float = 0.06) -> tuple[float, float, float, float]:
    points = np.asarray(points, dtype=np.float64)
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    span = float(max(high - low)) * (1.0 + pad)
    cx, cy = 0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])
    return cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2


def heat_grid(
    points: np.ndarray,
    values: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: int = 160,
    reach: float = 3.0,
) -> np.ma.MaskedArray:
    """Interpolate scattered values onto a grid, masking cells off the object."""
    points = np.asarray(points, dtype=np.float64)[:, :2]
    values = np.asarray(values, dtype=np.float64)
    xs = np.linspace(extent[0], extent[1], resolution)
    ys = np.linspace(extent[2], extent[3], resolution)
    gx, gy = np.meshgrid(xs, ys)
    linear = griddata(points, values, (gx, gy), method="linear")
    nearest = griddata(points, values, (gx, gy), method="nearest")
    filled = np.where(np.isnan(linear), nearest, linear)
    distance, _ = cKDTree(points).query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    cell = float(min(extent[1] - extent[0], extent[3] - extent[2])) / (resolution - 1)
    return np.ma.masked_array(filled, mask=distance.reshape(gx.shape) > reach * cell)


def _limits(per_action: Sequence[np.ndarray], relative: bool) -> tuple[float, float]:
    if relative:
        reach = max(float(np.percentile(np.abs(block), 99)) for block in per_action)
        return -reach, reach
    low = min(float(np.percentile(block, 2)) for block in per_action)
    high = max(float(np.percentile(block, 98)) for block in per_action)
    return low, high if high > low else low + 1.0


def plot_fim_heatmaps(
    nodes: np.ndarray,
    midpoints: np.ndarray,
    diagonals: Mapping[str, Mapping[str, np.ndarray]],
    names: Sequence[str],
    gains: Mapping[str, float],
    path: str | Path,
    title: str = "",
    selected: Sequence[str] = (),
    resolution: int = 160,
    relative: bool = False,
) -> Path:
    """One column per action, sorted by gain; rows are stiffness and mass.

    ``relative`` subtracts each action's own mean, so the panels show where the
    information sits rather than how large it is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = {"midpoints": np.asarray(midpoints, float), "nodes": np.asarray(nodes, float)}
    extent = square_extent(np.concatenate([points["midpoints"], points["nodes"]]))
    chosen = set(selected)

    values = {}
    for key, _, _ in ROWS:
        per_action = [
            np.log10(np.maximum(diagonals[name][key], 1e-30)) for name in names
        ]
        if relative:
            per_action = [column - column.mean() for column in per_action]
        values[key] = (per_action, _limits(per_action, relative))

    rows = len(ROWS)
    columns = len(names)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns + 1.4, 2.9 * rows + 0.9),
        squeeze=False,
    )

    for row, (key, source, label) in enumerate(ROWS):
        per_action, (low, high) = values[key]
        sample = points[source]
        image = None
        for column, name in enumerate(names):
            axis = axes[row, column]
            grid = heat_grid(sample, per_action[column], extent, resolution)
            image = axis.imshow(
                grid,
                extent=extent,
                origin="lower",
                cmap="coolwarm" if relative else "viridis",
                vmin=low,
                vmax=high,
            )
            axis.scatter(
                points["nodes"][:, 0], points["nodes"][:, 1],
                s=1.0, c="0.25", alpha=0.25, linewidths=0,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            if row == 0:
                color = "tab:orange" if name in chosen else "0.15"
                axis.set_title(f"{name}\nlogdet {gains[name]:.0f}", color=color, fontsize=9)
            if column == 0:
                axis.set_ylabel(label, fontsize=9)
        fig.colorbar(image, ax=axes[row, :].tolist(), shrink=0.82, pad=0.015)

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.93, top=0.86, bottom=0.04, wspace=0.06, hspace=0.14)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path
