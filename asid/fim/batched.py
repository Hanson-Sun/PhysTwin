"""One simulator instance running ``k`` rollouts side by side.

Instance ``i`` carries its own parameters, so a central-difference Jacobian is
one batched forward pass per chunk instead of ``2P`` serial rollouts. Action,
initial state and substepping are shared; only the parameter arrays differ.
Results are instance-major: object nodes are ``[instance 0 nodes][instance 1
nodes]...`` and the controller blocks follow the same layout.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from qqtt.utils import cfg
from qqtt.utils.mass_init import uniform_node_masses

# Like the serial harness: load the class with graph capture on, because the
# module switches Warp into debug codegen when the flag is off. The batched
# class itself never captures a graph.
cfg.use_graph = True
from qqtt.model.diff_simulator import SpringMassSystemWarp
from qqtt.model.diff_simulator.spring_mass_warp import SIM_FORCE_MODE_GATHER

cfg.use_graph = False

from asid.fim.case import Case
from asid.fim.simulator import SimConfig

wp.config.verbose = False
wp.config.verify_autograd_array_access = False


def build_signed_incidence_map(
    springs: np.ndarray, node_count: int, device: str
) -> tuple[torch.Tensor, int]:
    """Flattened signed spring ids per node, matching the trainer's map.

    Row ``n`` lists the springs touching node ``n``: ``+id`` adds the spring
    force and ``-id`` subtracts it. Ids are one-based so zero can pad a row.
    """
    pairs = torch.as_tensor(springs, dtype=torch.int64).cpu()
    incidence = [[] for _ in range(int(node_count))]
    for spring, (first, second) in enumerate(pairs.tolist()):
        spring_id = spring + 1
        if first < node_count:
            incidence[first].append(spring_id)
        if second < node_count:
            incidence[second].append(-spring_id)
    max_incident = max((len(entries) for entries in incidence), default=0)
    mapping = torch.zeros((int(node_count), max_incident), dtype=torch.int32)
    for node, entries in enumerate(incidence):
        if entries:
            mapping[node, : len(entries)] = torch.tensor(entries, dtype=torch.int32)
    return mapping.flatten().to(device=device), max_incident


def _tile(values: np.ndarray, instances: int) -> np.ndarray:
    return np.tile(values, (instances,) + (1,) * (values.ndim - 1))


class BatchedRollout:
    """``k`` finite-difference rollouts of the reduced case in one simulator."""

    def __init__(
        self, case: Case, config: SimConfig = SimConfig(), instances: int = 1
    ) -> None:
        if config.frames > case.frame_count:
            raise ValueError("requested frames exceed the case action")
        if instances < 1:
            raise ValueError("instances must be positive")
        self.case = case
        self.config = config
        self.instances = int(instances)
        self.sim = self._build(case, config, self.instances)

        device = cfg.device
        self._spring_Y_t = torch.empty(
            self.sim.n_springs_batched, dtype=torch.float32, device=device
        )
        self._masses_t = torch.empty(
            self.sim.object_massnode_total, dtype=torch.float32, device=device
        )
        self._dashpot_t = torch.empty(
            self.instances, dtype=torch.float32, device=device
        )
        self._spring_Y = wp.from_torch(self._spring_Y_t, requires_grad=False)
        self._masses = wp.from_torch(self._masses_t, requires_grad=False)
        self._dashpot = wp.from_torch(self._dashpot_t, requires_grad=False)
        self._start_t = torch.empty(
            self.sim.object_massnode_total, 3, dtype=torch.float32, device=device
        )
        self._start = wp.from_torch(self._start_t, requires_grad=False)
        self._controller_t = torch.empty(
            (config.frames, self.sim.num_controller_points, 3),
            dtype=torch.float32,
            device=device,
        )
        self._controller_wp = []
        self.set_action(case.controller, case.contact)

    @staticmethod
    def _build(
        case: Case, config: SimConfig, instances: int
    ) -> SpringMassSystemWarp:
        cfg.use_graph = False
        cfg.device = "cuda:0"
        cfg.data_type = "real"
        cfg.reverse_z = True
        cfg.controller_contact_radius = config.contact_radius
        cfg.controller_contact_activation_radius = config.contact_activation_radius
        cfg.controller_contact_release_radius = config.contact_release_radius
        # The serial harness passes stiffness explicitly; the batched class
        # reads it from the config and has no argument for it.
        cfg.controller_contact_stiffness = config.contact_stiffness

        nodes = len(case.nodes)
        # The serial ASID path contacts the dense controller-contact cloud; the
        # 30 pad points only drive its IK. The batched class has a single
        # controller block, so it must be the dense cloud: point contact against
        # the pad points alone moves the trajectory by 16 cm at 60 frames.
        controllers = len(case.contact[0])
        device = cfg.device
        vertices = np.concatenate(
            [_tile(case.nodes, instances), _tile(case.contact[0], instances)],
            axis=0,
        )
        incidence, max_incident = build_signed_incidence_map(
            case.springs, nodes, device
        )
        masses = uniform_node_masses(nodes, total_mass=config.total_mass)
        sim = SpringMassSystemWarp(
            base_springs=torch.tensor(
                case.springs, dtype=torch.int32, device=device
            ),
            base_rest_lengths=torch.tensor(case.rest_lengths, device=device),
            init_masses=torch.tensor(masses, device=device),
            init_masks=None,
            signed_incidence_map=incidence,
            max_incident_springs=max_incident,
            init_vertices=torch.tensor(vertices, device=device),
            init_velocities=None,
            dt=config.frame_dt / config.num_substeps,
            num_substeps=config.num_substeps,
            dashpot_damping=config.dashpot_damping,
            drag_damping=config.drag_damping,
            collision_dist=0.02,
            reverse_z=True,
            spring_Y_min=0.0,
            spring_Y_max=1e5,
            self_collision=False,
            collide_elas=torch.tensor(0.5, device=device),
            collide_fric=torch.tensor(0.5, device=device),
            collide_object_elas=torch.tensor(0.7, device=device),
            collide_object_fric=torch.tensor(0.3, device=device),
            spring_Y=torch.tensor(config.spring_Y, device=device),
            object_massnodes_total=instances * nodes,
            object_massnodes_single=nodes,
            controller_massnodes_single=controllers,
            controller_rest_location=torch.tensor(
                _tile(case.contact[0], instances), device=device
            ),
            number_of_instance=instances,
            controller_contact_friction=config.contact_friction,
            sim_force_mode=SIM_FORCE_MODE_GATHER,
            soft_ground=config.soft_ground,
            ground_stiffness=config.ground_stiffness,
            ground_damping=config.ground_damping,
            ground_smoothing=config.ground_smoothing,
            # The serial harness stores log stiffness and exponentiates in the
            # kernel; matching that keeps finite differences below the noise of
            # this discontinuous trajectory.
            spring_Y_log=True,
        )
        cfg.use_graph = False
        return sim

    def set_parameters(self, phi: np.ndarray, partition) -> None:
        """Write ``(k, P)`` grouped log parameters, one row per instance."""
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != (self.instances, partition.dimension):
            raise ValueError(
                f"phi must be ({self.instances}, {partition.dimension}), "
                f"got {phi.shape}"
            )
        spring_Y = np.stack([partition.spring_Y(row) for row in phi])
        masses = np.stack([partition.masses(row) for row in phi])
        dashpot = np.array([partition.dashpot(row) for row in phi])
        device = cfg.device
        scale = self.sim.mass_normalization
        # Same rounding order as Rollout.set_spring_Y: float32 values first,
        # then scale, then log. Neither path clamps after construction.
        self._spring_Y_t.copy_(
            torch.log(
                torch.as_tensor(spring_Y, dtype=torch.float32, device=device) * scale
            ).reshape(-1)
        )
        self._masses_t.copy_(
            torch.as_tensor(masses.reshape(-1), dtype=torch.float32, device=device)
        )
        self._dashpot_t.copy_(
            torch.as_tensor(dashpot * scale, dtype=torch.float32, device=device)
        )
        wp.copy(self.sim.wp_spring_Y_clamped, self._spring_Y)
        wp.copy(self.sim.wp_masses, self._masses)
        wp.copy(self.sim.wp_dashpot_damping, self._dashpot)
        wp.synchronize()

    def set_action(self, controller: np.ndarray, contact: np.ndarray) -> None:
        """Tile one ``(T, C, 3)`` contact trajectory into the ``k`` blocks.

        ``controller`` is the pad control cloud the serial class uses for IK;
        the batched class has no separate IK path, so only ``contact`` drives
        the simulator.
        """
        contact = np.asarray(contact)
        single = self.sim.controller_massnode_single
        if contact.ndim != 3 or contact.shape[1:] != (single, 3):
            raise ValueError(
                "contact point counts must match the simulator's geometry"
            )
        if len(contact) < self.config.frames:
            raise ValueError("action is shorter than the rollout")
        if not np.isfinite(contact).all():
            raise ValueError("action points must be finite")
        tiled = np.tile(contact[: self.config.frames], (1, self.instances, 1))
        self._controller_t.copy_(
            torch.as_tensor(tiled, dtype=torch.float32, device=cfg.device)
        )
        self._controller_wp = [
            wp.from_torch(self._controller_t[frame], dtype=wp.vec3, requires_grad=False)
            for frame in range(self.config.frames)
        ]

    def run(self, initial: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``(T, k*N, 3)`` node positions on the CPU."""
        sim = self.sim
        self._restart(initial)
        frames = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).clone()]
        for frame in range(1, self.config.frames):
            sim.set_controller_interactive(
                self._controller_wp[frame - 1], self._controller_wp[frame]
            )
            sim.step()
            frames.append(
                wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).clone()
            )
            # One copy each for x and v over the whole batch; a per-frame sync
            # here would erase the benefit of batching.
            wp.copy(sim.wp_states[0].wp_x, sim.wp_states[-1].wp_x)
            wp.copy(sim.wp_states[0].wp_v, sim.wp_states[-1].wp_v)
        return torch.stack(frames).cpu()

    def _restart(self, initial: torch.Tensor | None) -> None:
        """Load the first state, at rest, and reset contact activation."""
        sim = self.sim
        if initial is None:
            wp.copy(sim.wp_states[0].wp_x, sim.wp_init_vertices)
            wp.copy(sim.wp_states[0].wp_v, sim.wp_init_velocities)
        else:
            values = torch.as_tensor(
                initial, dtype=torch.float32, device=cfg.device
            )
            if values.shape != (len(self.case.nodes), 3):
                raise ValueError(f"initial state must be ({len(self.case.nodes)}, 3)")
            self._start_t.copy_(values.repeat(self.instances, 1))
            wp.copy(sim.wp_states[0].wp_x, self._start)
            sim.wp_states[0].wp_v.zero_()
        sim.controller_contact_active.zero_()
