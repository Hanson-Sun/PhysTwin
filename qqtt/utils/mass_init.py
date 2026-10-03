"""Physical initialization of spring-mass node masses.

Builders used to assign ``np.ones(n)`` -- 1.0 kg per node -- so an 11k-node
object weighed ~115 kN. Node mass sets the simulator's force scale: contact
normal force is ``K * penetration`` and the Coulomb friction limit is ``mu * N``,
both compared against the per-node weight ``m * g``. At 115 kN no contact
stiffness can hold the object, so friction grip is impossible.

The total matches the MuJoCo soft body that generated the data
(``mujoco_sim/soft_body.py`` DEFAULT_MASS) and is exposed as
``cfg.object_total_mass`` so a rigid case can override it.
"""

import numpy as np


def uniform_node_masses(n_object_nodes, total_length=None, total_mass=None):
    """Uniform per-node masses summing to the object's total mass.

    ``n_object_nodes`` counts OBJECT mass nodes only, so the object's weight
    does not depend on how many nodes the sampling produced. ``total_length``
    covers the full array when it also holds controller points, which the
    simulator never reads (``masses[local_idx]`` is object-local), so they are
    excluded from the total and are inert padding.
    """
    from qqtt.utils import cfg

    if total_mass is None:
        total_mass = float(getattr(cfg, "object_total_mass", 0.3))
    total_mass = float(total_mass)
    n_object_nodes = int(n_object_nodes)

    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError(
            f"object_total_mass must be a positive finite number, got {total_mass}"
        )
    if n_object_nodes <= 0:
        raise ValueError(f"n_object_nodes must be positive, got {n_object_nodes}")

    per_node = total_mass / n_object_nodes
    return np.full(
        n_object_nodes if total_length is None else int(total_length),
        per_node,
        dtype=np.float64,
    )