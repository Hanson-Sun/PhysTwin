"""Build the sketch dictionary for candidate pushes and greedily select a set."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from asid.fim.cache import ActionRunner, FactorCache, factor_key
from asid.fim.case import load_case
from asid.fim.fim import Prior, posterior_std
from asid.fim.group import build_partition
from asid.fim.logging_utils import Run, progress_printer
from asid.fim.operators import default_observation
from asid.fim.simulator import SimConfig
from asid.planning.actions import contact_frames, generate_actions, require_contact
from asid.planning.dictionary import SUBSPACES, build_dictionary, subspace_seed
from asid.planning.select import accumulate, greedy
from asid.planning.viz import plot_action_gains


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="data/different_types/sim_soft_sloth_push")
    parser.add_argument("--root", default="experiments-fim")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--groups", type=int, default=15)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=167)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--step", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--azimuth-count", type=int, default=6)
    parser.add_argument(
        "--tilts", type=float, nargs="+", default=[0.0],
        help="pitch in degrees; the family is every azimuth x tilt pair",
    )
    parser.add_argument(
        "--select", type=int, default=1, help="actions to keep; one per ASID iteration"
    )
    parser.add_argument("--subspace", choices=SUBSPACES, default="shared")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="factor cache; defaults to <root>/<case>/.factor_cache",
    )
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
    clearance = require_contact(case, actions, config.contact_radius)

    cache = FactorCache(args.cache_dir or Path(args.root) / case_name / ".factor_cache")

    def key(index: int, name: str) -> str:
        return factor_key(
            case,
            config,
            partition.phi0,
            actions[index].controller,
            actions[index].contact,
            rank,
            args.step,
            args.sigma,
            subspace_seed(args.seed, index, args.subspace),
        )

    started = time.time()
    runner = ActionRunner(case, config)
    requests = [
        (action.name, runner.run(action.controller, action.contact, partition, observation))
        for action in actions
    ]
    entries = build_dictionary(
        requests,
        partition.phi0,
        prior,
        args.sigma,
        rank=rank,
        step=args.step,
        seed=args.seed,
        subspace=args.subspace,
        progress=progress_printer(started),
        cache=cache,
        key=key,
    )
    selected = greedy(entries, args.select)
    factors = None
    for entry, _ in selected:
        factors = accumulate(factors, entry.U)
    information = np.zeros((partition.dimension, partition.dimension))
    if factors is not None:
        information += factors @ factors.T
    posterior = posterior_std(prior, np.eye(partition.dimension) + information)

    payload = {
        "actions": [
            {
                "name": entry.name,
                "gain": entry.gain,
                "azimuth_deg": float(np.rad2deg(action.azimuth)),
                "tilt_deg": float(np.rad2deg(action.tilt)),
                "clearance": clearance[entry.name],
                "contact_frames": contact_frames(
                    case.nodes, action.contact, config.contact_radius
                ),
            }
            for entry, action in zip(entries, actions)
        ],
        "selected": [
            {"name": entry.name, "marginal_gain": gain} for entry, gain in selected
        ],
        "rank": rank,
        "dimension": int(partition.dimension),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }

    with Run.create(args.root, case_name, run_id, vars(args)) as experiment:
        for entry, action in zip(entries, actions):
            directory = f"actions/{entry.name}"
            experiment.save_array(f"{directory}/controller.npy", action.controller, "controller")
            experiment.save_array(f"{directory}/contact.npy", action.contact, "contact")
            experiment.save_array(f"{directory}/U.npy", entry.U, "sketch factor")
            experiment.save_array(f"{directory}/JQ.npy", entry.JQ, "sketch projection")
        experiment.save_json("gains.json", payload)
        experiment.save_array("posterior_std.npy", posterior, "posterior std")
        figure = plot_action_gains(
            case.nodes,
            partition.node_groups,
            actions,
            [entry.gain for entry in entries],
            [entry.name for entry, _ in selected],
            experiment.path / "figs" / "action_gains.png",
            title=(
                f"{case_name}  g={args.groups}  frames={args.frames}  "
                f"rank={rank}"
            ),
            contact_radius=config.contact_radius,
        )
    print(cache.report())
    print(f"run: {Path(args.root) / case_name / run_id}")
    print(f"figure: {figure}")
    print(f"selected: {[entry.name for entry, _ in selected]}")


if __name__ == "__main__":
    main()
