"""Exact Fisher gains for a saved action set, compared with the sketch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from asid.fim.case import load_case
from asid.fim.fim import Fisher, Prior
from asid.fim.group import build_partition
from asid.fim.operators import default_observation, jacobian, parameterized_run
from asid.fim.simulator import Rollout, SimConfig
from asid.fim.sketch import random_subspace
from asid.planning.dictionary import Entry
from asid.planning.select import greedy, subspace_entry
from asid.planning.viz import plot_gain_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--select", type=int, default=None)
    parser.add_argument("--reuse-fisher", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run
    config = json.loads((run_dir / "config.json").read_text())
    gains = json.loads((run_dir / "gains.json").read_text())
    names = [action["name"] for action in gains["actions"]]
    sketch_gains = {action["name"]: action["gain"] for action in gains["actions"]}
    select_count = args.select or len(gains["selected"])

    case = load_case(
        config["case"], num_nodes=config["nodes"], max_frames=config["frames"]
    )
    sim_config = SimConfig(
        frames=config["frames"], num_substeps=config["substeps"]
    )
    partition = build_partition(
        case, config["groups"], seed=config["seed"], config=sim_config
    )
    observation = default_observation(config["frames"], config["stride"], config["sigma"])
    prior = Prior().std(partition.groups)

    started = time.time()
    # One simulator for every action: set_action swaps the trajectory and run()
    # restarts the state, so rebuilding per action only repeats kernel setup.
    rollout = Rollout(case, sim_config)
    exact_entries = []
    exact_gains = {}
    whitened = {}
    for done, name in enumerate(names, start=1):
        directory = run_dir / "actions" / name
        matrix_path = directory / "fisher.npy"
        if args.reuse_fisher and matrix_path.exists():
            fisher = Fisher(np.load(matrix_path), prior)
        else:
            rollout.set_action(
                np.load(directory / "controller.npy"), np.load(directory / "contact.npy")
            )
            run = parameterized_run(rollout, partition, observation)
            jac = jacobian(run, partition.phi0, config["step"], central=True)
            fisher = Fisher.from_jacobian(jac, config["sigma"], prior)
            np.save(matrix_path, fisher.matrix)
        eigenvalues, vectors = np.linalg.eigh(fisher.whitened)
        U = vectors * np.sqrt(np.maximum(eigenvalues, 0.0))
        exact_gains[name] = fisher.logdet()
        exact_entries.append(Entry(name, fisher.logdet(), U))
        whitened[name] = fisher.whitened

        elapsed = time.time() - started
        print(
            f"[{done}/{len(names)}] {name} exact={exact_gains[name]:.3f} "
            f"sketch={sketch_gains[name]:.3f} {elapsed / 60:.1f} min",
            flush=True,
        )

    exact_values = [exact_gains[name] for name in names]
    sketch_values = [sketch_gains[name] for name in names]
    exact_selected = greedy(exact_entries, select_count)
    # ``selected`` is the pipeline's decision, which is exact-based once
    # --exact-top is used; ``sketch_selected`` is what the screen alone picked.
    sketch_selected = [
        name for name in gains.get("sketch_selected", [])
    ] or [entry["name"] for entry in gains["selected"]]
    exact_top = names[int(np.argmax(exact_values))]
    sketch_top = names[int(np.argmax(sketch_values))]
    exact_names = [entry.name for entry, _ in exact_selected]

    Q = random_subspace(len(prior), gains["rank"], config["seed"])
    shared = [subspace_entry(name, whitened[name], Q) for name in names]
    shared_joint = sum(gain for _, gain in greedy(shared, len(names)))
    exact_joint = sum(gain for _, gain in greedy(exact_entries, len(names)))

    payload = {
        "exact": exact_gains,
        "sketch": sketch_gains,
        "spearman": float(spearmanr(sketch_values, exact_values).statistic),
        "top1_match": sketch_top == exact_top,
        "exact_top": exact_top,
        "sketch_top": sketch_top,
        "exact_selection": exact_names,
        "sketch_selection": sketch_selected,
        "selection_match": exact_names == sketch_selected,
        "marginal_gains": [
            {"name": entry.name, "gain": gain} for entry, gain in exact_selected
        ],
        "shared_subspace": {
            "rank": int(gains["rank"]),
            "joint_gain": shared_joint,
            "exact_joint_gain": exact_joint,
            "capture": shared_joint / max(exact_joint, 1e-30),
            "selection": [entry.name for entry, _ in greedy(shared, select_count)],
        },
    }
    (run_dir / "comparison.json").write_text(json.dumps(payload, indent=2))
    figure = plot_gain_comparison(
        names,
        sketch_values,
        exact_values,
        run_dir / "figs" / "action_ranking.png",
        title=(
            f"{Path(config['case']).name}  g={config['groups']}  "
            f"rank={gains['rank']}  spearman={payload['spearman']:.3f}"
        ),
    )
    print(json.dumps(payload, indent=2))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
