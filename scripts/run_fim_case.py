"""Build the exact grouped Fisher on the reduced case and plot the overlay."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from asid.fim.case import load_case
from asid.fim.fim import Fisher, Prior
from asid.fim.group import build_partition
from asid.fim.logging_utils import Run, progress_printer
from asid.fim.operators import default_observation, jacobian, parameterized_run
from asid.fim.simulator import Rollout, SimConfig
from asid.fim.viz import (
    plot_fim_matrix,
    plot_fim_overlay,
    plot_point_information,
    point_information,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="data/different_types/sim_soft_sloth_push")
    parser.add_argument("--root", default="experiments-fim")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=167)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--step", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-step-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_name = Path(args.case).name
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    config = SimConfig(frames=args.frames, num_substeps=args.substeps)
    case = load_case(args.case, num_nodes=args.nodes, max_frames=args.frames)
    partition = build_partition(case, args.groups, seed=args.seed, config=config)
    observation = default_observation(args.frames, args.stride, args.sigma)
    rollout = Rollout(case, config)
    run = parameterized_run(rollout, partition, observation)
    phi0 = partition.phi0
    prior = Prior().std(partition.groups)

    started = time.time()
    jac = jacobian(run, phi0, args.step, central=True, progress=progress_printer(started))
    elapsed = time.time() - started
    fisher = Fisher.from_jacobian(jac, args.sigma, prior)
    diagonal = np.diag(fisher.matrix)
    point_values = point_information(jac, prior, observation.sigma, len(case.nodes))

    summary = {
        "case": case_name,
        "nodes": args.nodes,
        "groups": args.groups,
        "dimension": int(partition.dimension),
        "frames": args.frames,
        "stride": args.stride,
        "sigma": args.sigma,
        "step": args.step,
        "central": True,
        "soft_ground": config.soft_ground,
        "rollout_seconds": elapsed / (2 * partition.dimension),
        "logdet": fisher.logdet(),
        "eigenvalues": fisher.eigenvalues().tolist(),
        "dashpot_log10_fim": float(np.log10(diagonal[partition.indices["c"]][0])),
    }

    if not args.skip_step_check:
        check = np.linspace(0, partition.dimension - 1, 3, dtype=int)
        second = jacobian(
            run, phi0, 2.0 * args.step, indices=check, central=True
        )
        relative = np.linalg.norm(second - jac[check], axis=1) / np.maximum(
            np.linalg.norm(jac[check], axis=1), 1e-30
        )
        summary["step_check_indices"] = check.tolist()
        summary["step_check_relative"] = relative.tolist()

    midpoints = case.object_midpoints

    with Run.create(args.root, case_name, run_id, vars(args)) as experiment:
        experiment.save_array("nodes.npy", case.nodes, "object nodes")
        experiment.save_array("node_groups.npy", partition.node_groups, "node group id")
        experiment.save_array(
            "node_centroids.npy", partition.node_centroids, "group centroids"
        )
        experiment.save_array("spring_groups.npy", partition.spring_groups, "spring group id")
        experiment.save_array("midpoints.npy", midpoints, "spring midpoints")
        experiment.save_array("phi0.npy", phi0, "nominal phi")
        experiment.save_array("fim_diag.npy", diagonal, "diag(F)")
        experiment.save_array("fisher.npy", fisher.matrix, "F")
        experiment.save_array("point_information.npy", point_values, "standalone per-node local Gaussian gain in nats")
        experiment.save_array("eigenvalues.npy", fisher.eigenvalues(), "whitened eigenvalues")
        experiment.save_array(
            "posterior_std.npy", fisher.posterior_std(), "posterior std"
        )
        experiment.save_json("summary.json", summary)
        plot_fim_matrix(fisher.matrix, experiment.path / "figs" / "fim_matrix.png")
        plot_point_information(
            case.nodes, point_values, experiment.path / "figs" / "point_information.png"
        )
        figure = plot_fim_overlay(
            case.nodes,
            midpoints,
            partition.node_groups,
            partition.spring_groups,
            diagonal,
            partition.indices,
            experiment.path / "figs" / "fim_partition.png",
            title=f"{case_name}  g={args.groups}  frames={args.frames}",
        )
    print(f"run: {Path(args.root) / case_name / run_id}")
    print(f"figure: {figure}")
    print(f"logdet: {summary['logdet']:.3f}")


if __name__ == "__main__":
    main()
