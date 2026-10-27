"""CEM search over spline tool motions inside an :class:`~asid.planning.envelope.Envelope`.

The score is a log-determinant, and elite ranking is scale-free, so CEM needs no
rescaling. The screened rigid action supplies the starting mean, so the plan can
never score below the discrete baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from asid.planning import spline
from asid.planning.cem import CemResult, cem
from asid.planning.envelope import Diagnostics, Envelope
from asid.planning.spline import Motion, Spec


@dataclass(frozen=True)
class Plan:
    """The searched motion: its spec, its sampled controls, and its search trace."""

    spec: Spec
    motion: Motion
    result: CemResult
    diagnostics: Diagnostics


def plan_spline(
    score: Callable[[np.ndarray, np.ndarray], float],
    template: tuple[np.ndarray, np.ndarray],
    envelope: Envelope,
    *,
    knots: int = 3,
    samples: int = 8,
    iterations: int = 3,
    elites: int = 2,
    seed: int = 0,
) -> Plan:
    """Search the spline controls of one tool motion with CEM.

    ``score`` maps a sampled ``(controller, contact)`` pair to an information
    gain; ``envelope`` owns every constraint, and a rejected candidate is dropped
    rather than penalised.
    """

    def objective(vector: np.ndarray) -> float | None:
        motion = envelope.admit(spline.unflatten(vector, knots), template)
        return None if motion is None else float(score(motion.controller, motion.contact))

    result = cem(
        objective,
        spline.flatten(Spec.start(knots)),
        *envelope.bounds(knots),
        samples=samples,
        iterations=iterations,
        elites=elites,
        project=lambda vector: spline.flatten(spline.unflatten(vector, knots)),
        seed=seed,
    )
    spec = spline.unflatten(result.vector, knots)
    motion = envelope.admit(spec, template)
    if motion is None:  # the search only ever returns an evaluated candidate
        raise RuntimeError("searched spec violates the envelope it was searched in")
    return Plan(spec, motion, result, envelope.diagnostics(motion))
