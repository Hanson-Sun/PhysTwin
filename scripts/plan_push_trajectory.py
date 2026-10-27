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
import dataclasses
import json
from pathlib import Path

import numpy as np

from asid.fim.cache import ExactCache, FactorCache
from asid.fim.case import load_case
from asid.fim.fim import Prior
from asid.fim.group import build_partition
from asid.fim.operators import default_observation
from asid.fim.simulator import SimConfig
from asid.planning.actions import generate_actions, require_contact
from asid.planning.envelope import Envelope
from asid.planning.measure import Measure
from asid.planning.search import plan_spline


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

    current = None
    if args.current_factors is not None:
        current = np.load(args.current_factors)["U"]

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
        factors=FactorCache(
            args.cache_dir or Path(args.root) / case_dir.name / ".factor_cache"
        ),
        exacts=ExactCache(
            args.exact_cache or Path(args.root) / case_dir.name / ".exact_cache"
        ),
        current=current,
    )

    azimuths = np.linspace(0.0, 360.0, args.azimuth_count, endpoint=False)
    actions = generate_actions(case, azimuths, tilts=args.tilts)
    require_contact(case, actions, config.contact_radius)

    entries = measure.screen(actions)
    decision = measure.decide(entries, actions, 1, args.exact_top)
    start_entry = {entry.name: entry for entry in entries}[decision.chosen[0].name]
    start = {action.name: action for action in actions}[start_entry.name]
    print(
        f"screened {len(actions)} actions ({measure.factors.report()}); "
        f"initializer basis: {decision.basis}",
        flush=True,
    )
    for entry in sorted(entries, key=lambda item: item.gain, reverse=True):
        exact = decision.exact.get(entry.name)
        note = f"  exact {exact:7.2f}" if exact is not None else ""
        print(f"  {entry.name:<10} sketch {entry.gain:7.2f}{note}", flush=True)

    envelope = Envelope.from_demo(
        case.nodes,
        (start.controller, start.contact),
        frames=args.frames,
        dt=config.frame_dt,
        contact_radius=config.contact_radius,
        margin=args.margin,
        offset_limit=args.offset_limit,
        orientation_limit=np.deg2rad(args.orientation_limit),
    )
    plan = plan_spline(
        measure.score,
        (start.controller, start.contact),
        envelope,
        knots=args.knots,
        samples=args.samples,
        iterations=args.iterations,
        elites=args.elites,
        seed=args.seed,
    )
    print(
        f"CEM from {start_entry.name}: "
        + " -> ".join(f"{value:.2f}" for value in plan.result.history),
        flush=True,
    )
    print(
        f"plan score {plan.result.score:.2f} over {plan.result.evaluations} "
        f"evaluations; {plan.diagnostics.summary()}",
        flush=True,
    )

    exact_scores: dict[str, float] = {}
    if not args.skip_exact_score:
        exact_scores["start"] = measure.exact(
            start_entry.name, start.controller, start.contact
        )
        exact_scores["plan"] = measure.exact(
            "plan", plan.motion.controller, plan.motion.contact
        )
        exact_scores["delta"] = exact_scores["plan"] - exact_scores["start"]
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
        "selection_basis": decision.basis,
        "exact": decision.exact,
        "exact_scores": exact_scores,
        "score": plan.result.score,
        "score_basis": f"rank-{measure.rank} sketch, "
        + ("accumulated factors" if current is not None else "prior factors only"),
        "history": plan.result.history,
        "evaluations": plan.result.evaluations,
        "spec": {
            "positions": plan.spec.positions.tolist(),
            "times": plan.spec.times.tolist(),
            "yaw": plan.spec.yaw.tolist(),
            "pitch": plan.spec.pitch.tolist(),
        },
        "samples": {
            "offsets": plan.motion.positions.tolist(),
            "yaw": plan.motion.yaw.tolist(),
            "pitch": plan.motion.pitch.tolist(),
        },
        "diagnostics": dataclasses.asdict(plan.diagnostics),
        "knots": args.knots,
        "rank": measure.rank,
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
