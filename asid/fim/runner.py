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


class BatchedActionRunner:
    """Reuse one batched simulator, returning ``phi_batch -> observations``.

    ``k`` is the chunk the caller runs; the rollout is rebuilt when that size
    changes, so a chunked Jacobian allocates one simulator per distinct chunk
    (in practice two: the full chunks and the remainder).
    """

    def __init__(self, case, config):
        self.case = case
        self.config = config
        self.rollout = None
        self.instances = None

    def _get(self, instances: int):
        from asid.fim.batched import BatchedRollout  # keeps this module Warp-free

        if self.rollout is None or self.instances != instances:
            self.rollout = BatchedRollout(self.case, self.config, instances)
            self.instances = instances
        return self.rollout

    def run(
        self, controller, contact, partition, observation, initial=None
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return ``phi_batch -> (k, D) observations`` for one trajectory."""

        def evaluate(phi_batch):
            rollout = self._get(len(phi_batch))
            rollout.set_action(controller, contact)
            rollout.set_parameters(phi_batch, partition)
            return observation.batch(rollout.run(initial=initial), len(phi_batch))

        return evaluate
