"""Action gains and the approach direction of each candidate push."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asid.planning.actions import Action, minimum_clearance


def _first_contact_patch(
    nodes: np.ndarray, contact: np.ndarray, radius: float
) -> np.ndarray | None:
    """Centroid of the nodes touched at the first contact frame."""
    frames = np.asarray(contact, dtype=np.float64)
    hits = np.flatnonzero(minimum_clearance(nodes, frames) <= radius)
    if not hits.size:
        return None
    points = np.asarray(nodes, dtype=np.float64)
    distance = np.linalg.norm(points[None] - frames[hits[0]][:, None], axis=-1)
    return points[distance.min(axis=0) <= radius].mean(axis=0)


def plot_action_gains(
    nodes: np.ndarray,
    node_groups: np.ndarray,
    actions: Sequence[Action],
    gains: Sequence[float],
    selected: Sequence[str],
    path: str | Path,
    title: str = "",
    contact_radius: float = 0.02,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = int(node_groups.max()) + 1
    chosen = set(selected)

    fig, (map_axis, bar_axis) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )
    map_axis.scatter(
        nodes[:, 0], nodes[:, 1],
        c=node_groups, cmap="turbo", vmin=0, vmax=max(groups - 1, 1), s=8,
    )
    for action, gain in zip(actions, gains):
        patch = _first_contact_patch(nodes, action.contact, contact_radius)
        if patch is None:
            continue
        is_selected = action.name in chosen
        map_axis.scatter(
            patch[0], patch[1],
            marker="X", s=90 if is_selected else 45,
            c="tab:orange" if is_selected else "tab:blue",
            edgecolors="black", linewidths=0.5, zorder=3,
        )
        map_axis.annotate(
            f"{action.name} {gain:.1f}",
            (patch[0], patch[1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7 if is_selected else 6,
        )
    map_axis.set_aspect("equal")
    map_axis.set_title("first contact patch (top view)")
    map_axis.set_xticks([])
    map_axis.set_yticks([])

    order = np.argsort(gains)[::-1]
    names = [actions[i].name for i in order]
    colors = ["tab:orange" if name in chosen else "tab:blue" for name in names]
    bar_axis.bar(range(len(order)), [gains[i] for i in order], color=colors)
    bar_axis.set_xticks(range(len(order)))
    bar_axis.set_xticklabels(names, rotation=45, ha="right")
    bar_axis.set_ylabel("log det(I + H_a)")
    bar_axis.set_title("standalone gain (orange = selected)")
    bar_axis.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.18, wspace=0.18)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sequential_scores(
    rounds: Sequence[dict], path: str | Path, title: str = ""
) -> Path:
    """Conditioned gain per candidate, one panel per round."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(rounds), figsize=(3.4 * len(rounds) + 0.6, 3.2), squeeze=False
    )
    for axis, item in zip(axes[0], rounds):
        scores = item["scores"]
        chosen = item["selected"]
        names = sorted(scores, key=lambda name: scores[name])
        colors = ["tab:orange" if name == chosen else "tab:blue" for name in names]
        axis.barh(names, [scores[name] for name in names], color=colors)
        axis.set_title(
            f"round {item['round']}: {chosen}\n{scores[chosen]:.1f} nats", fontsize=9
        )
        axis.set_xlabel(r"$\log\det(I + P^{-1} H_a)$", fontsize=8)
        axis.grid(axis="x", alpha=0.3)
        axis.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gain_comparison(
    names: Sequence[str],
    sketch: Sequence[float],
    exact: Sequence[float],
    path: str | Path,
    title: str = "",
) -> Path:
    """Sketch gain against the exact gain, one point per action."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(exact, sketch, s=45)
    for name, x, y in zip(names, exact, sketch):
        axis.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    low = min(min(exact), min(sketch))
    high = max(max(exact), max(sketch))
    axis.plot([low, high], [low, high], color="0.5", lw=0.8, ls="--")
    axis.set_xlabel("exact log det(I + H_a)")
    axis.set_ylabel("sketch gain")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
