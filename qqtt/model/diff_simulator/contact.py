"""Shared tangential friction primitives for Warp contact kernels."""

import warp as wp


@wp.func
def elastic_stick_slip(
    displacement: wp.vec3,
    tangential_velocity: wp.vec3,
    dt: float,
    tangential_stiffness: float,
    friction_limit: float,
):
    """Return differentiable friction force and updated bristle displacement.

    Tangential displacement accumulates while sticking, allowing friction at
    zero relative velocity. The elastic force is smoothly capped by the single
    Coulomb limit ``mu*N`` and the stored displacement is return-mapped to that
    cap during sliding.
    """
    trial_displacement = displacement + dt * tangential_velocity
    trial_force = -tangential_stiffness * trial_displacement
    trial_magnitude = wp.length(trial_force)
    stiffness = wp.max(tangential_stiffness, 1e-8)

    friction_force = trial_force
    if friction_limit <= 1e-8:
        friction_force = wp.vec3(0.0, 0.0, 0.0)
    elif trial_magnitude > friction_limit:
        friction_force = trial_force * (friction_limit / trial_magnitude)
    next_displacement = -friction_force / stiffness
    return friction_force, next_displacement
