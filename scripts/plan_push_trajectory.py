"""Plan one spline tool motion for the next ASID iteration.

Screens the azimuth/tilt push family, then refines the winner with CEM over a
spline: waypoint offsets, a time warp and a claw orientation. The result is
replayed by ``generate_sim_data.py --trajectory``.

    python scripts/plan_push_trajectory.py --case data/different_types/sim_soft_sloth_push \
        --out experiments-fim/sim_soft_sloth_push/trajectory.json

The objective is the rank-``rank`` sketch gain at the nominal grouped parameters,
optionally conditioned on accumulated factors (``--current-factors``). Gradients
are unavailable -- the Warp rollout is built with backward disabled -- so the
search is CEM over sampled candidates, and every candidate pays its own rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asid.fim.cache import (
    ActionRunner,
    ExactCache,
    FactorCache,
    exact_gain,
    exact_key,
    factor_key,
)
from asid.fim.case import load_case
from asid.fim.fim import Prior
from asid.fim.group import build_partition
from asid.fim.operators import default_observation
from asid.fim.simulator import SimConfig
from asid.fim.sketch import gain_against, randomized_sketch
from asid.planning.actions import Action, generate_actions, require_contact
from asid.planning.dictionary import build_dictionary
from asid.planning.select import exact_ordered, shortlist
from asid.planning.spline import controls, plan_spline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="data/different_types/sim_soft_sloth_push")
    parser.add_argument("--root", default="experiments-fim")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--substeps", type=int, default=167)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--step", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--azimuth-count", type=int, default=6)
    parser.add_argument("--tilts", type=float, nargs="+", default=[0.0])
    parser.add_argument("--knots", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--offset-limit", type=float, default=0.05, help="metres")
    parser.add_argument("--orientation-limit", type=float, default=30.0, help="degrees")
    parser.add_argument("--margin", type=float, default=0.15, help="workspace padding, metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--exact-top",
        type=int,
        default=1,
        help=(
            "screen the family, then compute the exact Fisher for its best N actions "
            "and start CEM from the exact-best; 0 starts from the sketch-best"
        ),
    )
    parser.add_argument("--exact-cache", type=Path, default=None)
    parser.add_argument(
        "--skip-exact-score",
        action="store_true",
        help="skip the exact Fisher score of the plan and of its initializer",
    )
    parser.add_argument("--current-factors", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_dir = Path(args.case)
    config = SimConfig(
        frames=args.frames, num_substeps=args.substeps, frame_dt=1.0 / 30.0
    )
    case = load_case(case_dir, num_nodes=args.nodes, max_frames=args.frames)
    partition = build_partition(case, args.groups, config=config)
    observation = default_observation(args.frames, args.stride, args.sigma)
    prior = Prior().std(partition.groups)
    rank = min(args.rank, partition.dimension)
    runner = ActionRunner(case, config)
    cache = FactorCache(args.cache_dir or Path(args.root) / case_dir.name / ".factor_cache")

    azimuths = np.linspace(0.0, 360.0, args.azimuth_count, endpoint=False)
    actions = generate_actions(case, azimuths, tilts=args.tilts)
    require_contact(case, actions, config.contact_radius)

    def key(index: int, name: str | None = None) -> str:
        return factor_key(
            case,
            config,
            partition.phi0,
            actions[index].controller,
            actions[index].contact,
            rank,
            args.step,
            args.sigma,
            args.seed,
        )

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
        cache=cache,
        key=key,
    )
    by_name = {action.name: action for action in actions}
    exact = ExactCache(
        args.exact_cache or Path(args.root) / case_dir.name / ".exact_cache"
    )

    def exact_of(action: Action) -> float:
        """Exact Fisher log-det of a screened action, cached by its own bytes."""
        action_key = exact_key(
            case,
            config,
            partition.phi0,
            action.controller,
            action.contact,
            args.step,
            args.sigma,
        )
        cached = exact.get(action_key)
        if cached is None:
            print(
                f"exact Fisher {action.name}: {2 * partition.dimension + 1} rollouts",
                flush=True,
            )
            cached = exact_gain(
                runner.run(action.controller, action.contact, partition, observation),
                partition.phi0,
                args.step,
                args.sigma,
                prior,
            )
            exact.put(action_key, *cached)
        return float(cached[0])

    exact_gains: dict[str, float] = {}
    if args.exact_top > 0:
        candidates = shortlist(entries, min(args.exact_top, len(entries)))
        for entry in candidates:
            exact_gains[entry.name] = exact_of(by_name[entry.name])
        start_entry = exact_ordered(candidates, exact_gains)[0]
        basis = f"exact Fisher over the best {len(candidates)} of {len(entries)} screened"
    else:
        start_entry = max(entries, key=lambda item: item.gain)
        basis = f"rank-{rank} sketch over {len(entries)} screened"
    start = by_name[start_entry.name]
    print(f"screened {len(actions)} actions ({cache.report()}); initializer basis: {basis}", flush=True)
    for entry in sorted(entries, key=lambda item: item.gain, reverse=True):
        note = f"  exact {exact_gains[entry.name]:7.2f}" if entry.name in exact_gains else ""
        print(f"  {entry.name:<10} sketch {entry.gain:7.2f}{note}", flush=True)

    current = None
    if args.current_factors is not None:
        current = np.load(args.current_factors)["U"]
        if current.ndim != 2 or current.shape[0] != partition.dimension:
            raise ValueError("current factors must be (dimension, k) in phi coordinates")

    def score(controller: np.ndarray, contact: np.ndarray) -> float:
        run = runner.run(controller, contact, partition, observation)
        sketch = randomized_sketch(
            run, partition.phi0, prior, args.sigma, rank, args.step, args.seed
        )
        return gain_against(sketch.U, current)

    # The workspace is what the demonstration already occupies plus a margin: an
    # object-only box would reject the recorded approach itself (the claw starts
    # half a stroke away from the object).
    occupied = np.concatenate(
        [case.nodes, start.controller.reshape(-1, 3), start.contact.reshape(-1, 3)]
    )
    peak = max(
        float(np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=-1).max())
        for points in (start.controller, start.contact)
    )
    plan = plan_spline(
        score,
        (start.controller, start.contact),
        case.nodes,
        frames=args.frames,
        lower=occupied.min(axis=0) - args.margin,
        upper=occupied.max(axis=0) + args.margin,
        max_speed=1.5 * peak / config.frame_dt,
        dt=config.frame_dt,
        contact_radius=config.contact_radius,
        knots=args.knots,
        offset_limit=args.offset_limit,
        samples=args.samples,
        iterations=args.iterations,
        elites=args.elites,
        seed=args.seed,
        orientation_limit=np.deg2rad(args.orientation_limit),
    )
    offsets, yaw, pitch = controls(plan.spec, args.frames)
    print(f"CEM from {start_entry.name}: " + " -> ".join(
        f"{value:.2f}" for value in plan.result.history
    ), flush=True)
    print(
        f"plan score {plan.result.score:.2f} over {plan.result.evaluations} evaluations; "
        f"max offset {np.abs(offsets).max():.3f} m, yaw {np.abs(yaw).max():.2f} rad, "
        f"pitch {np.abs(pitch).max():.2f} rad",
        flush=True,
    )

    exact_scores: dict[str, float] = {}
    if not args.skip_exact_score:
        exact_scores["start"] = (
            exact_gains[start_entry.name] if start_entry.name in exact_gains else exact_of(start)
        )
        plan_gain, _ = exact_gain(
            runner.run(plan.controller, plan.contact, partition, observation),
            partition.phi0,
            args.step,
            args.sigma,
            prior,
        )
        exact_scores["plan"] = float(plan_gain)
        exact_scores["delta"] = float(plan_gain) - exact_scores["start"]
        print(
            f"exact Fisher: initializer {exact_scores['start']:.2f} -> plan "
            f"{exact_scores['plan']:.2f} ({exact_scores['delta']:+.2f})",
            flush=True,
        )

    payload = {
        "case": case_dir.name,
        "start_action": start_entry.name,
        "start_gain": start_entry.gain,
        "screen": {entry.name: entry.gain for entry in entries},
        "selection_basis": basis,
        "exact": exact_gains,
        "exact_scores": exact_scores,
        "score": plan.result.score,
        "score_basis": "rank-{} sketch, prior factors only".format(rank),
        "history": plan.result.history,
        "evaluations": plan.result.evaluations,
        "spec": {
            "positions": plan.spec.positions.tolist(),
            "times": plan.spec.times.tolist(),
            "yaw": plan.spec.yaw.tolist(),
            "pitch": plan.spec.pitch.tolist(),
        },
        "samples": {
            "offsets": offsets.tolist(),
            "yaw": yaw.tolist(),
            "pitch": pitch.tolist(),
        },
        "knots": args.knots,
        "rank": rank,
        "groups": args.groups,
        "frames": args.frames,
        "nodes": args.nodes,
        "basis": "nominal phi0, prior factors only"
        if current is None
        else "nominal phi0 with accumulated factors",
    }
    out = args.out or Path(args.root) / case_dir.name / "trajectory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
