"""Rank and power-iteration study for the randomized sketch.

Reads the exact per-action Fisher saved by ``run_fim_case.py`` under
``experiments-fim/<case>/<run_id>/actions/*/fisher.npy`` and reports what a
sketch of each configuration would have scored.

Because ``H`` is already known, only the *subspace* is under test:

    gain(H, Q) = log det(I + Q^T H Q),   Q = orth(H^q Omega)

with ``Omega`` a Rademacher matrix. That is what a perfect ``J`` and ``J^T``
would give, so it is an upper bound on a real sketch, which also pays
finite-difference noise. ``q`` power iterations cost ``2 * rank * (q + 1)``
rollouts per action: every iteration needs a ``J^T`` application, which this
harness cannot provide (it steps the simulator with ``requires_grad=False``),
so ``q > 0`` is measurable here but not available in ``asid/fim/sketch.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from asid.fim.fim import Prior
from asid.planning.dictionary import Entry
from asid.planning.select import greedy, subspace_entry

DEFAULT_RUNS = (
    ("g10", "mix_g10_r12", 10),
    ("g20", "mix_g20_r12", 20),
)


def load_case(root: Path, run_id: str, groups: int) -> dict[str, np.ndarray]:
    """Whitened exact Fisher per action, keyed by action name."""
    prior_std = Prior().std(groups)
    actions = root / run_id / "actions"
    out = {}
    for action in sorted(actions.iterdir()):
        path = action / "fisher.npy"
        if path.is_file():
            out[action.name] = np.load(path) * prior_std[:, None] * prior_std[None, :]
    if not out:
        raise FileNotFoundError(f"no exact fisher.npy under {actions}")
    return out


def exact_gain(matrix: np.ndarray) -> float:
    return float(np.linalg.slogdet(np.eye(len(matrix)) + matrix)[1])


def subspace_gain(matrix: np.ndarray, Q: np.ndarray) -> float:
    return float(np.linalg.slogdet(np.eye(Q.shape[1]) + Q.T @ matrix @ Q)[1])


def exact_entry(name: str, matrix: np.ndarray) -> Entry:
    """Factor of the full Fisher, for the reference selection."""
    values, vectors = np.linalg.eigh(matrix)
    return Entry(name, 0.0, vectors * np.sqrt(np.maximum(values, 0.0)))


def random_basis(dimension: int, rank: int, seed: int) -> np.ndarray:
    omega = np.random.default_rng(seed).choice([-1.0, 1.0], size=(dimension, rank))
    return np.linalg.qr(omega)[0]


def projected_basis(matrix: np.ndarray, rank: int, q: int, seed: int) -> np.ndarray:
    """``orth(H^q Omega)``: a range finder with ``q`` power iterations."""
    projected = np.random.default_rng(seed).choice(
        [-1.0, 1.0], size=(len(matrix), rank)
    )
    for _ in range(q):
        projected = matrix @ projected
    return np.linalg.qr(projected)[0]


def sketch_entries(
    fishers: dict[str, np.ndarray], rank: int, q: int, shared: bool, seed: int = 0
) -> list[Entry]:
    """Factors of each action's Fisher restricted to a rank-``rank`` subspace.

    ``shared`` picks one subspace from ``seed`` for every action, which is what
    ``build_dictionary`` does by default; the alternative gives each action its
    own, so a union of factors is not capped at ``rank`` dimensions.
    """
    out = []
    for index, (name, matrix) in enumerate(fishers.items()):
        action_seed = seed if shared else seed + 1 + index
        Q = projected_basis(matrix, rank, q, action_seed)
        out.append(subspace_entry(name, matrix, Q))
    return out


def greedy_names(entries: list[Entry], count: int) -> list[str]:
    return [entry.name for entry, _ in greedy(entries, count)]


def recorded_gains(run_dir: Path) -> dict[str, float]:
    data = json.load(open(run_dir / "gains.json"))
    return {action["name"]: action["gain"] for action in data["actions"]}


def report_validation(
    run_dir: Path, fishers: dict[str, np.ndarray], rank: int
) -> None:
    """Does the offline model reproduce the sketch that was actually recorded?"""
    recorded = recorded_gains(run_dir)
    offline = {e.name: e.gain for e in sketch_entries(fishers, rank, 0, shared=True)}
    rows = [(n, offline[n], recorded[n]) for n in fishers if n in recorded]
    print(f"\n[1] offline (rank {rank}, q=0) vs recorded sketch gain")
    for name, mine, theirs in rows:
        print(f"    {name:<14} offline {mine:8.2f}  recorded {theirs:8.2f}"
              f"  {mine / theirs - 1:+7.1%}")
    if len(rows) > 1:
        print(f"    spearman {spearmanr([r[1] for r in rows], [r[2] for r in rows]).statistic:.3f}"
              f", mean |diff| {np.mean([abs(r[1] / r[2] - 1) for r in rows]):.1%}"
              "  (finite-difference noise in the recorded sketch)")


def report_capture(
    fishers: dict[str, np.ndarray],
    exact: dict[str, float],
    ranks: list[int],
    seeds: int,
) -> None:
    print("\n[2] capture = sketch gain / exact gain, over "
          f"{seeds} seeds x {len(fishers)} actions")
    print(f"    {'rank':>4} | {'q=0':>18} {'q=1':>18} {'q=2':>18} | {'rollouts/action':>15}")
    for rank in ranks:
        cells = []
        for q in (0, 1, 2):
            ratios = [e.gain / exact[e.name]
                      for seed in range(seeds)
                      for e in sketch_entries(fishers, rank, q, shared=False, seed=seed)]
            cells.append(f"{np.mean(ratios):7.1%} (min {np.min(ratios):5.1%})")
        print(f"    {rank:>4} | " + " ".join(f"{c:>18}" for c in cells) +
              f" | {2 * rank * 3:>15}")


def report_budget(
    fishers: dict[str, np.ndarray],
    exact: dict[str, float],
    budgets: list[int],
    seeds: int,
) -> None:
    """Same rollout budget, split between rank and power iterations."""
    dimension = len(next(iter(fishers.values())))
    print("\n[3] best capture at an equal rollout budget"
          " (rank = budget / (2 * (q + 1)), capped at P)")
    print(f"    {'budget':>6} | {'q=0':>10} {'q=1':>10} {'q=2':>10} | winner")
    for budget in budgets:
        row = {}
        for q in (0, 1, 2):
            rank = budget // (2 * (q + 1))
            if not 2 <= rank <= dimension:
                continue
            ratios = [e.gain / exact[e.name]
                      for seed in range(seeds)
                      for e in sketch_entries(fishers, rank, q, shared=False, seed=seed)]
            row[q] = (np.mean(ratios), rank)
        if not row:
            continue
        best = max(row, key=lambda q: row[q][0])
        cells = [f"{row[q][0]:9.1%}" if q in row else f"{'-':>9}" for q in (0, 1, 2)]
        print(f"    {budget:>6} | " + " ".join(cells) +
              f" | q={best} at rank {row[best][1]}")


def report_ordering(
    fishers: dict[str, np.ndarray],
    exact: dict[str, float],
    ranks: list[int],
    seeds: int,
    count: int,
) -> None:
    """How small the subspace can get before the ranking breaks."""
    names = list(fishers)
    exact_order = greedy_names(
        [exact_entry(n, fishers[n]) for n in names], count
    )
    print(f"\n[4] ordering at q=0, {seeds} seeds, against the exact greedy set")
    print(f"    {'rank':>4} | {'spearman':>16} {'top-1':>6} {'greedy set':>10}")
    for rank in ranks:
        spears, tops, hits = [], [], []
        for seed in range(seeds):
            entries = sketch_entries(fishers, rank, 0, shared=True, seed=seed)
            got = {e.name: e.gain for e in entries}
            spears.append(spearmanr([got[n] for n in names],
                                    [exact[n] for n in names]).statistic)
            tops.append(max(names, key=got.get) == max(names, key=exact.get))
            hits.append(greedy_names(entries, count) == exact_order)
        print(f"    {rank:>4} | {np.mean(spears):.3f} (worst {np.min(spears):.3f})"
              f" {np.mean(tops):>6.0%} {np.mean(hits):>10.0%}")


def report_accumulation(
    fishers: dict[str, np.ndarray],
    rank: int,
    count: int,
) -> None:
    """How much joint information the accumulated factors actually carry."""
    dimension = len(next(iter(fishers.values())))
    order = greedy_names([exact_entry(n, fishers[n]) for n in fishers], count)
    print(f"\n[5] accumulated gain over the exact greedy order {order}")
    print(f"    {'k':>2} | {'exact':>9} {'shared':>9} {'indep':>9} | "
          f"{'shared/exact':>12} {'indep/exact':>11}")
    for k in range(1, count + 1):
        chosen = order[:k]
        exact_total = float(np.linalg.slogdet(
            np.eye(dimension) + sum(fishers[n] for n in chosen))[1])
        cells = {}
        for shared in (True, False):
            entries = {e.name: e for e in sketch_entries(fishers, rank, 0, shared)}
            block = np.concatenate([entries[n].U for n in chosen], axis=1)
            cells[shared] = float(np.linalg.slogdet(
                np.eye(dimension) + block @ block.T)[1])
        print(f"    {k:>2} | {exact_total:>9.1f} {cells[True]:>9.1f} "
              f"{cells[False]:>9.1f} | {cells[True] / exact_total:>11.1%} "
              f"{cells[False] / exact_total:>10.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="experiments-fim")
    parser.add_argument("--case", default="sim_soft_sloth_push")
    parser.add_argument("--rank", type=int, default=12, help="the operating rank")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--select", type=int, default=3)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[24, 48, 72, 96]
    )
    args = parser.parse_args()

    root = Path(args.root) / args.case
    for label, run_id, groups in DEFAULT_RUNS:
        fishers = load_case(root, run_id, groups)
        exact = {name: exact_gain(matrix) for name, matrix in fishers.items()}
        dimension = len(next(iter(fishers.values())))
        print(f"\n{'=' * 78}\n{label}  {run_id}   P={dimension}   "
              f"{len(fishers)} actions")
        print("  exact gain: " + "  ".join(f"{n}={exact[n]:.1f}" for n in fishers))

        ranks = sorted({r for r in (4, 6, 8, 12, 16, 20, 24, 32, 40, dimension)
                        if 2 <= r <= dimension})
        report_validation(root / run_id, fishers, args.rank)
        report_capture(fishers, exact, ranks, args.seeds)
        report_budget(fishers, exact, args.budgets, args.seeds)
        report_ordering(fishers, exact, ranks, args.seeds, args.select)
        report_accumulation(fishers, args.rank, args.select)


if __name__ == "__main__":
    main()
