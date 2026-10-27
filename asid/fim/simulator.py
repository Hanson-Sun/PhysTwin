"""Warp rollout harness for the reduced spring-mass case.

Release codegen, direct stepping, persistent state buffers. Graph capture is
skipped because it crashes in the contact-grid radix sort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from qqtt.utils import cfg

# The upstream module is imported with the flag on: when it is off the module
# switches Warp into debug codegen, and when it stays on the constructor
# captures a CUDA graph, which fails in the contact-grid radix sort.
cfg.use_graph = True
from qqtt.model.diff_simulator.spring_mass_warp_upstream import SpringMassSystemWarp
from qqtt.utils.mass_init import uniform_node_masses

from asid.fim.case import Case

cfg.use_graph = False
wp.config.verbose = False
wp.config.verify_autograd_array_access = False


@dataclass(frozen=True)
class SimConfig:
    frames: int = 60
    num_substeps: int = 167
    frame_dt: float = 1.0 / 30.0
    spring_Y: float = 3e4
    dashpot_damping: float = 100.0
    total_mass: float = 0.3
    drag_damping: float = 3.0
    contact_radius: float = 0.020
    contact_activation_radius: float = 0.020
    contact_release_radius: float = 0.025
    contact_stiffness: float = 2000.0
    contact_friction: float = 0.3
    soft_ground: bool = True
    ground_stiffness: float = 1000.0
    ground_damping: float = 2.0
    ground_smoothing: float = 1e-4


class Rollout:
    """A single finite-difference rollout of the reduced case."""

    def __init__(self, case: Case, config: SimConfig = SimConfig()) -> None:
        if config.frames > case.frame_count:
            raise ValueError("requested frames exceed the case action")
        self.case = case
        self.config = config
        self.sim = self._build(case, config)
        self._carry_x = wp.empty_like(self.sim.wp_states[0].wp_x, requires_grad=False)
        self._carry_v = wp.empty_like(self.sim.wp_states[0].wp_v, requires_grad=False)
        self._spring_Y_t = torch.empty(
            self.sim.n_springs, dtype=torch.float32, device=cfg.device
        )
        self._masses_t = torch.empty(
            self.sim.num_object_points, dtype=torch.float32, device=cfg.device
        )
        self._dashpot_t = torch.empty(1, dtype=torch.float32, device=cfg.device)
        self._spring_Y = wp.from_torch(self._spring_Y_t, requires_grad=False)
        self._masses = wp.from_torch(self._masses_t, requires_grad=False)
        self._dashpot = wp.from_torch(self._dashpot_t, requires_grad=False)
        self._start_t = torch.empty(
            len(case.nodes), 3, dtype=torch.float32, device=cfg.device
        )
        self._start = wp.from_torch(self._start_t, requires_grad=False)

    @staticmethod
    def _build(case: Case, config: SimConfig) -> SpringMassSystemWarp:
        cfg.use_graph = False
        cfg.device = "cuda:0"
        cfg.data_type = "real"
        cfg.reverse_z = True
        previous = (
            getattr(cfg, "learn_damping", False),
            getattr(cfg, "learn_mass", False),
        )
        cfg.learn_damping = False
        cfg.learn_mass = False
        cfg.controller_contact_radius = config.contact_radius
        cfg.controller_contact_activation_radius = config.contact_activation_radius
        cfg.controller_contact_release_radius = config.contact_release_radius

        nodes = len(case.nodes)
        vertices = np.concatenate([case.nodes, case.controller[0]], axis=0)
        masses = uniform_node_masses(nodes, total_mass=config.total_mass)
        dummy_points = torch.zeros(2, nodes, 3, device=cfg.device)
        dummy_mask = torch.zeros(2, nodes, dtype=torch.bool, device=cfg.device)
        sim = SpringMassSystemWarp(
            init_vertices=torch.tensor(vertices, device=cfg.device),
            init_springs=torch.tensor(case.springs, dtype=torch.int32, device=cfg.device),
            init_rest_lengths=torch.tensor(case.rest_lengths, device=cfg.device),
            init_masses=torch.tensor(masses, device=cfg.device),
            dt=config.frame_dt / config.num_substeps,
            num_substeps=config.num_substeps,
            spring_Y=config.spring_Y,
            collide_elas=0.5,
            collide_fric=0.5,
            dashpot_damping=config.dashpot_damping,
            drag_damping=config.drag_damping,
            collide_object_elas=0.7,
            collide_object_fric=0.3,
            init_masks=None,
            collision_dist=0.02,
            init_velocities=None,
            num_object_points=nodes,
            num_surface_points=nodes,
            num_original_points=nodes,
            controller_points=torch.tensor(case.controller, device=cfg.device),
            controller_contact_points=torch.tensor(case.contact, device=cfg.device),
            controller_contact_stiffness=config.contact_stiffness,
            controller_contact_friction=config.contact_friction,
            learn_controller_contact=False,
            reverse_z=True,
            spring_Y_min=0.0,
            spring_Y_max=1e5,
            gt_object_points=dummy_points,
            gt_object_visibilities=dummy_mask,
            gt_object_motions_valid=dummy_mask,
            self_collision=False,
            disable_backward=True,
            deterministic_forces=True,
            soft_ground=config.soft_ground,
            ground_stiffness=config.ground_stiffness,
            ground_damping=config.ground_damping,
            ground_smoothing=config.ground_smoothing,
        )
        cfg.learn_damping, cfg.learn_mass = previous
        return sim

    def set_spring_Y(self, values: torch.Tensor) -> None:
        values = torch.as_tensor(values, dtype=torch.float32, device=cfg.device)
        if values.numel() == 1:
            values = values.repeat(self.sim.n_springs)
        self._spring_Y_t.copy_(torch.log(values * self.sim.mass_normalization))
        wp.copy(self.sim.wp_spring_Y, self._spring_Y)
        wp.synchronize()

    def set_masses(self, values: torch.Tensor) -> None:
        values = torch.as_tensor(values, dtype=torch.float32, device=cfg.device)
        self._masses_t.copy_(values)
        wp.copy(self.sim.wp_masses, self._masses)
        wp.synchronize()

    def set_dashpot(self, value: float) -> None:
        self._dashpot_t.fill_(value * self.sim.mass_normalization)
        wp.copy(self.sim.wp_dashpot_damping, self._dashpot)
        wp.synchronize()

    def set_action(self, controller: np.ndarray, contact: np.ndarray) -> None:
        controller = np.asarray(controller)
        contact = np.asarray(contact)
        if (controller.ndim != 3 or controller.shape[1:] != (self.sim.num_control_points, 3)
            or contact.ndim != 3 or contact.shape[1:] != (self.sim.num_contact_points, 3)):
            raise ValueError("action point counts must match the simulator's controller geometry")
        if len(controller) != len(contact):
            raise ValueError("controller and contact trajectories disagree")
        if len(controller) < self.config.frames:
            raise ValueError("action is shorter than the rollout")
        if not np.isfinite(controller).all() or not np.isfinite(contact).all():
            raise ValueError("action points must be finite")
        self.sim.controller_points = torch.as_tensor(controller, dtype=torch.float32, device=cfg.device)
        self.sim.controller_contact_points = torch.as_tensor(contact, dtype=torch.float32, device=cfg.device)
        motion = torch.linalg.vector_norm(
            self.sim.controller_contact_points[1:]
            - self.sim.controller_contact_points[:-1],
            dim=-1,
        )
        self.sim.controller_contact_sweep_radius = float(
            motion.amax() / max(self.sim.num_substeps, 1)
        )

    def run(self, initial: torch.Tensor | None = None) -> torch.Tensor:
        """Simulate the rollout, optionally starting from a deformed state.

        ``initial`` is the node positions to start from, at rest. Without it the
        rollout starts from the case's initial state.
        """
        sim = self.sim
        self._restart(initial)
        wp.synchronize()

        frames = [wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).clone()]
        for frame in range(1, self.config.frames):
            sim.set_controller_target(frame, pure_inference=True)
            sim.step()
            wp.synchronize()
            frames.append(wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).clone())
            wp.copy(self._carry_x, sim.wp_states[-1].wp_x)
            wp.copy(self._carry_v, sim.wp_states[-1].wp_v)
            wp.copy(sim.wp_states[0].wp_x, self._carry_x)
            wp.copy(sim.wp_states[0].wp_v, self._carry_v)
            wp.synchronize()
        return torch.stack(frames).cpu()

    def _restart(self, initial: torch.Tensor | None) -> None:
        """Load the first state, at rest, and reset contact activation."""
        if initial is None:
            wp.copy(self._carry_x, self.sim.wp_init_vertices)
            wp.copy(self._carry_v, self.sim.wp_init_velocities)
        else:
            values = torch.as_tensor(initial, dtype=torch.float32, device=cfg.device)
            if values.shape != (len(self.case.nodes), 3):
                raise ValueError(f"initial state must be ({len(self.case.nodes)}, 3)")
            self._start_t.copy_(values)
            wp.copy(self._carry_x, self._start)
            self._carry_v.zero_()
        wp.copy(self.sim.wp_states[0].wp_x, self._carry_x)
        wp.copy(self.sim.wp_states[0].wp_v, self._carry_v)
        self.sim.controller_contact_active.zero_()
