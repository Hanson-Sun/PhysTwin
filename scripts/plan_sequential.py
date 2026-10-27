"""Sequential action selection: pick each push given what the previous ones taught.

Candidates are re-simulated from the deformed state left by the chosen pushes,
and scored by ``log det(I + P^{-1} H_a)`` with ``P`` the accumulated posterior
precision. The parameter mean stays at the assumed truth; only the posterior
precision and the body state carry over.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from asid.fim.case import load_case
from asid.fim.fim import Prior, posterior_std
from asid.fim.group import build_partition
from asid.fim.logging_utils import Run
from asid.fim.operators import default_observation, parameterized_run
from asid.fim.simulator import Rollout, SimConfig
from asid.fim.sketch import randomized_sketch
from asid.planning.actions import generate_actions, require_contact
from asid.planning.sequential import marginal_gain
from asid.planning.viz import plot_sequential_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="data/different_types/sim_soft_sloth_push")
    parser.add_argument("--root", default="experiments-fim")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=167)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--step", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--azimuth-count", type=int, default=6)
    parser.add_argument("--tilts", type=float, nargs="+", default=[0.0])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_name = Path(args.case).name
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    config = SimConfig(frames=args.frames, num_substeps=args.substeps)
    case = load_case(args.case, num_nodes=args.nodes, max_frames=args.frames)
    partition = build_partition(case, args.groups, seed=args.seed, config=config)
    observation = default_observation(args.frames, args.stride, args.sigma)
    prior = Prior().std(partition.groups)
    rank = min(args.rank, partition.dimension)

    azimuths = np.linspace(0.0, 360.0, args.azimuth_count, endpoint=False)
    actions = generate_actions(case, azimuths, tilts=args.tilts)
    require_contact(case, actions, config.contact_radius)

    started = time.time()
    remaining = list(actions)
    precision_matrix = np.eye(partition.dimension)
    state: np.ndarray | None = None
    rounds: list[dict] = []
    saved: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for round_index in range(1, args.rounds + 1):
        if not remaining:
            break
        scores, factors = {}, {}
        for done, action in enumerate(remaining, start=1):
            rollout = Rollout(case, config)
            rollout.set_action(action.controller, action.contact)
            run = parameterized_run(rollout, partition, observation, initial=state)
            sketch = randomized_sketch(
                run,
                partition.phi0,
                prior,
                args.sigma,
                rank=rank,
                step=args.step,
                seed=args.seed,
            )
            factors[action.name] = sketch.U
            scores[action.name] = marginal_gain(
                precision_matrix, sketch.U @ sketch.U.T
            )
            elapsed = (time.time() - started) / 60.0
            print(
                f"round {round_index} [{done}/{len(remaining)}] {action.name} "
                f"gain {scores[action.name]:.2f}  {elapsed:.1f} min",
                flush=True,
            )

        best = max(scores, key=lambda name: scores[name])
        chosen = next(action for action in remaining if action.name == best)
        rollout = Rollout(case, config)
        rollout.set_action(chosen.controller, chosen.contact)
        partition.apply(rollout, partition.phi0)
        trajectory = rollout.run(initial=state).numpy()
        state = trajectory[-1]
        precision_matrix = precision_matrix + factors[best] @ factors[best].T
        remaining = [action for action in remaining if action.name != best]
        saved[best] = (factors[best], chosen.controller, chosen.contact)

        std = posterior_std(prior, precision_matrix)
        rounds.append(
            {
                "round": round_index,
                "scores": scores,
                "selected": best,
                "selected_prior_gain": scores[best],
                "posterior_std": {
                    "k": std[partition.indices["k"]].tolist(),
                    "c": std[partition.indices["c"]].tolist(),
                    "m": std[partition.indices["m"]].tolist(),
                },
            }
        )
        print(f"round {round_index} selected {best}  ({scores[best]:.2f})", flush=True)

    payload = {
        "rounds": rounds,
        "selected": [item["selected"] for item in rounds],
        "rank": rank,
        "dimension": int(partition.dimension),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    with Run.create(args.root, case_name, run_id, vars(args)) as run:
        run.save_json("rounds.json", payload)
        for index, item in enumerate(rounds, start=1):
            name = item["selected"]
            factors, controller, contact = saved[name]
            directory = f"rounds/r{index}_{name}"
            run.save_array(f"{directory}/U.npy", factors, "sketch factor")
            run.save_array(f"{directory}/controller.npy", controller, "controller")
            run.save_array(f"{directory}/contact.npy", contact, "contact")
        run.save_array(
            "posterior_std.npy", posterior_std(prior, precision_matrix), "posterior std"
        )
        run.save_array("final_state.npy", state, "body state after the last push")
        figure = plot_sequential_scores(
            rounds,
            run.path / "figs" / "sequential_scores.png",
            title=f"{case_name}  g={args.groups}  rank={rank}  sequential",
        )

    print(f"selected: {payload['selected']}")
    print(f"run: {Path(args.root) / case_name / run_id}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
