"""Compare the q=0 sketch with a saved exact Fisher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asid.fim.case import load_case
from asid.fim.fim import Fisher, Prior
from asid.fim.group import build_partition
from asid.fim.operators import default_observation, parameterized_run
from asid.fim.simulator import Rollout, SimConfig
from asid.fim.sketch import randomized_sketch
from asid.fim.validation import spectrum_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.run
    config = json.loads((run_dir / "config.json").read_text())
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
    rollout = Rollout(case, sim_config)
    evaluate = parameterized_run(rollout, partition, observation)
    prior = Prior().std(partition.groups)

    fisher = Fisher(np.load(run_dir / "fisher.npy"), prior)
    sketch = randomized_sketch(
        evaluate,
        partition.phi0,
        prior,
        config["sigma"],
        rank=args.rank,
        step=config["step"],
        seed=args.seed,
    )
    payload = {
        "rank": args.rank,
        "exact_logdet": fisher.logdet(),
        "sketch_gain": sketch.gain,
        "gain_ratio": sketch.gain / max(fisher.logdet(), 1e-30),
        **spectrum_summary(fisher.eigenvalues(), args.rank),
    }
    np.save(run_dir / "sketch_U.npy", sketch.U)
    (run_dir / "sketch_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
