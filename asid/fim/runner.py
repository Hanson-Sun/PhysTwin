"""One simulator instance, re-armed for each trajectory under test.

``Rollout`` compiles Warp kernels and allocates its graph, so building one per
candidate costs seconds each. ``set_action`` swaps the controller and contact
trajectories and ``run`` restarts from the case state, so a single instance
serves a whole search as long as evaluation stays sequential.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class ActionRunner:
    """Reuse one rollout for many trajectories, returning ``phi -> observation``."""

    def __init__(self, case, config):
        from asid.fim.simulator import Rollout  # keeps this module Warp-free

        self.rollout = Rollout(case, config)

    def run(
        self, controller, contact, partition, observation, initial=None
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return ``phi -> observation vector`` for one trajectory."""

        def evaluate(phi):
            self.rollout.set_action(controller, contact)
            partition.apply(self.rollout, phi)
            return observation.vector(self.rollout.run(initial=initial))

        return evaluate
