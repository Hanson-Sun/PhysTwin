"""Small sampled-FIM-to-MuJoCo smoke test; not a calibrated sim2sim benchmark.

Run in WSL with the project environment:
    conda run -n phystwin-cu132 env PYTHONPATH=. python scripts/plan_mujoco_smoke.py

The two finger-pad centres approximate the tool; the initial object geometry
comes from a reconstructed case and is not an exact MuJoCo state match.
"""

from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from asid.fim.case import build_controller_tethers, load_case
from asid.fim.group import build_partition
from asid.fim.operators import default_observation
from asid.fim.simulator import Rollout, SimConfig
from asid.planning.mujoco import execute
from asid.planning.trajectory import fisher_score, plan_trajectory, rigid_template
from mujoco_sim.scene import load_model
from mujoco_sim.simulation import DigitalTwinSim


def main() -> None:
    case = load_case("data/different_types/sim_soft_sloth_push", num_nodes=200, max_frames=6)
    model = load_model(
        1, object_mesh="object_sloth.stl",
        soft={"rgba": "0.47 0.42 0.37 1", "young": 350, "cellcount": "4 6 1"},
    )
    sim = DigitalTwinSim(model, width=16, height=12)
    tree = cKDTree(case.nodes)
    z0 = float(np.median(case.nodes[:, 2]))
    choices = np.flatnonzero(np.abs(case.nodes[:, 2] - z0) < 0.025)
    target = case.nodes[choices[np.argmin(case.nodes[choices, 0])]]
    initial = np.array([target[0] - 0.05, target[1], -target[2] + 0.029])
    quat = np.array([1.0, 0, 0, 0])
    sim.set_interactor_pose("interactor0", initial, quat)
    mujoco.mj_forward(sim.model, sim.data)
    ids = [model.geom("interactor0_pad_l").id, model.geom("interactor0_pad_r").id]
    reflection = np.array([1.0, 1.0, -1.0])
    points = np.asarray(sim.data.geom_xpos)[ids] * reflection
    print("initial pad-node distances (m):", tree.query(points)[0], flush=True)
    positions = initial * reflection + np.column_stack(
        (np.linspace(0, 0.02, 6), np.zeros(6), np.zeros(6))
    )
    controller, contact = rigid_template(points, points, positions)
    tethers, tether_rest = build_controller_tethers(case.nodes, controller[0], 0.04, 50, len(case.nodes))
    case = replace(
        case, controller=controller.astype(np.float32), contact=contact.astype(np.float32),
        springs=np.concatenate((case.springs[:case.num_object_springs], tethers)),
        rest_lengths=np.concatenate((case.rest_lengths[:case.num_object_springs], tether_rest)),
    )
    dt = 30 * float(model.opt.timestep)
    config = SimConfig(frames=6, num_substeps=100, frame_dt=dt)
    partition = build_partition(case, 2, config=config)
    observation = default_observation(6, 1, 0.005)
    rollout = Rollout(case, config)
    prior = np.array([0.5, 0.5, 0.5, 0.3, 0.3])

    def score(trajectory):
        return fisher_score(
            rollout, partition, observation, trajectory, [partition.phi0], np.ones(1),
            prior, None, 2, 0.01, 0,
        )

    plan = plan_trajectory(
        controller, contact, score, lower=np.full(3, -2.0), upper=np.full(3, 2.0),
        max_speed=0.25, dt=dt, offset_limit=0.002, knots=1, candidates=3, seed=2,
    )
    result = execute(sim, plan.trajectory, initial_position=initial, quaternion=quat, frame_dt=dt)
    print("sampled scores (nats):", plan.sampled_scores)
    print("selected score and controls:", plan.score, plan.controls)
    print("RGB-D frames, commands, elapsed:", len(result.frames), len(result.commands), result.times[-1])
    if len(result.frames) != 5 or len(result.commands) != 150 or not np.isfinite(plan.score):
        raise RuntimeError("sim2sim smoke test did not produce the expected result")


if __name__ == "__main__":
    main()
