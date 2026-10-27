"""Sketch screen over candidate pushes, then the one action the iteration acts on."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from asid.fim.cache import ExactCache, FactorCache
from asid.fim.case import load_case
from asid.fim.fim import Prior, posterior_std
from asid.fim.group import build_partition
from asid.fim.logging_utils import Run, progress_printer
from asid.fim.operators import default_observation
from asid.fim.simulator import SimConfig
from asid.planning.actions import contact_frames, generate_actions, require_contact
from asid.planning.dictionary import SUBSPACES
from asid.planning.measure import Measure
from asid.planning.select import accumulate
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
        "--tilts",
        type=float,
        nargs="+",
        default=[0.0],
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
    parser.add_argument(
        "--exact-top",
        type=int,
        default=1,
        help=(
            "compute the exact Fisher for the best N screened actions and select on "
            "that; 0 keeps pure sketch selection. The sketch's finite-difference "
            "protocol differs from the exact one, so on close candidates it can "
            "rank them the wrong way round."
        ),
    )
    parser.add_argument(
        "--exact-cache",
        type=Path,
        default=None,
        help="exact Fisher cache; defaults to <root>/<case>/.exact_cache",
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

    azimuths = np.linspace(0.0, 360.0, args.azimuth_count, endpoint=False)
    actions = generate_actions(case, azimuths, tilts=args.tilts)
    clearance = require_contact(case, actions, config.contact_radius)

    started = time.time()
    measure = Measure(
        case,
        config,
        partition,
        observation,
        prior,
        rank=args.rank,
        step=args.step,
        sigma=args.sigma,
        seed=args.seed,
        subspace=args.subspace,
        factors=FactorCache(args.cache_dir or Path(args.root) / case_name / ".factor_cache"),
        exacts=ExactCache(args.exact_cache or Path(args.root) / case_name / ".exact_cache"),
    )
    entries = measure.screen(actions, progress=progress_printer(started))
    decision = measure.decide(entries, actions, args.select, args.exact_top)
    note = f"{measure.exacts.report()}; " if decision.exact else ""
    print(f"{note}{decision.basis}: {[choice.name for choice in decision.chosen]}")

    by_entry = {entry.name: entry for entry in entries}
    by_action = {action.name: action for action in actions}
    factors = None
    for choice in decision.chosen:
        factors = accumulate(factors, by_entry[choice.name].U)
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
        **decision.payload(),
        "exact_top": args.exact_top,
        "rank": measure.rank,
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
        for name in decision.exact:
            action = by_action[name]
            experiment.save_array(
                f"actions/{name}/fisher.npy",
                measure.matrix(action.controller, action.contact),
                "exact Fisher",
            )
        experiment.save_json("gains.json", payload)
        experiment.save_array("posterior_std.npy", posterior, "posterior std")
        figure = plot_action_gains(
            case.nodes,
            partition.node_groups,
            actions,
            [entry.gain for entry in entries],
            [choice.name for choice in decision.chosen],
            experiment.path / "figs" / "action_gains.png",
            title=(
                f"{case_name}  g={args.groups}  frames={args.frames}  "
                f"rank={measure.rank}"
            ),
            contact_radius=config.contact_radius,
        )
    print(measure.factors.report())
    print(f"run: {Path(args.root) / case_name / run_id}")
    print(f"figure: {figure}")
    print(f"selected: {[choice.name for choice in decision.chosen]}")


if __name__ == "__main__":
    main()
