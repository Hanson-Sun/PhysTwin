"""Gain of a candidate trajectory, and the choice it drives.

One owner for the measurement chain the planner repeats: the reused rollout, the
sketch screen, the exact arbitration, the caches behind both, and the decision a
comparison run reads back. Everything except the trajectory is fixed per ASID
iteration, so a :class:`Measure` is built once and asked about candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from asid.fim.cache import ExactCache, FactorCache, exact_key, factor_key
from asid.fim.fim import Fisher
from asid.fim.operators import jacobian
from asid.fim.runner import ActionRunner
from asid.fim.sketch import gain_against, randomized_sketch
from asid.planning.dictionary import Entry, build_dictionary, subspace_seed
from asid.planning.select import exact_ordered, greedy, shortlist


@dataclass(frozen=True)
class Choice:
    """One candidate and the gain that justified it."""

    name: str
    gain: float


@dataclass(frozen=True)
class Decision:
    """Which candidate the next iteration acts on, and what the screen wanted.

    ``chosen`` is the decision (exact-arbitrated when a shortlist was measured),
    ``screened`` is what the sketch alone would have picked. Both are recorded:
    the screen still proposes, so its disagreement with the exact metric is a
    result, not something to reconstruct after the fact.
    """

    chosen: list[Choice]
    screened: list[str]
    exact: dict[str, float]
    basis: str
    sketch_top: str
    exact_top: str | None

    @property
    def agreement(self) -> bool:
        return self.exact_top is not None and self.exact_top == self.sketch_top

    def payload(self) -> dict:
        """The fields written to ``gains.json`` and read back by comparisons."""
        return {
            "selected": [
                {"name": choice.name, "marginal_gain": choice.gain}
                for choice in self.chosen
            ],
            "sketch_selected": list(self.screened),
            "exact": dict(self.exact),
            "selection_basis": self.basis,
            "rank_agreement": {
                "sketch_top": self.sketch_top,
                "exact_top": self.exact_top,
                "top1_match": self.agreement,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "Decision":
        """Read a recorded decision.

        Runs written before exact arbitration carried no ``sketch_selected``
        field; for those the screen's pick is the recorded selection.
        """
        chosen = [
            Choice(item["name"], float(item.get("marginal_gain", 0.0)))
            for item in payload["selected"]
        ]
        screened = list(
            payload.get("sketch_selected") or [choice.name for choice in chosen]
        )
        agreement = payload.get("rank_agreement") or {}
        return cls(
            chosen=chosen,
            screened=screened,
            exact=dict(payload.get("exact") or {}),
            basis=payload.get("selection_basis", ""),
            sketch_top=agreement.get("sketch_top"),
            exact_top=agreement.get("exact_top"),
        )


def exact_fisher(
    run: Callable[[np.ndarray], np.ndarray],
    phi0: np.ndarray,
    step: float,
    sigma: float,
    prior_std: np.ndarray,
) -> Fisher:
    """Exact Fisher of a ``phi -> observation`` map at ``phi0``."""
    return Fisher.from_jacobian(
        jacobian(run, phi0, step, central=True), sigma, prior_std
    )


def arbitrate(
    entries: Sequence[Entry],
    count: int,
    exact_top: int,
    exact_of: Callable[[str], float],
    rank: int,
) -> Decision:
    """Pick ``count`` candidates: exact Fisher over a sketch shortlist.

    The sketch's finite-difference protocol differs from the exact one, so at
    ``g = 20`` it inverted close candidates (az270 233.23 > az090 230.75 by
    sketch against 628.37 < 692.67 exactly). ``exact_top = 0`` keeps pure sketch
    selection for cheap runs.
    """
    screened = [entry.name for entry, _ in greedy(entries, count)]
    sketch_top = max(entries, key=lambda entry: entry.gain).name
    if exact_top <= 0:
        return Decision(
            chosen=[Choice(entry.name, gain) for entry, gain in greedy(entries, count)],
            screened=screened,
            exact={},
            basis=f"rank-{rank} sketch",
            sketch_top=sketch_top,
            exact_top=None,
        )
    candidates = shortlist(entries, min(exact_top, len(entries)))
    gains = {entry.name: float(exact_of(entry.name)) for entry in candidates}
    ranked = exact_ordered(candidates, gains)
    return Decision(
        chosen=[Choice(entry.name, gains[entry.name]) for entry in ranked[:count]],
        screened=screened,
        exact=gains,
        basis=f"exact Fisher over the best {len(candidates)} of {len(entries)} screened",
        sketch_top=sketch_top,
        exact_top=ranked[0].name,
    )


class Measure:
    """Gain of one trajectory under the iteration's fixed settings."""

    def __init__(
        self,
        case,
        config,
        partition,
        observation,
        prior_std: np.ndarray,
        *,
        rank: int,
        step: float,
        sigma: float,
        seed: int = 0,
        subspace: str = "shared",
        factors: FactorCache | None = None,
        exacts: ExactCache | None = None,
        current: np.ndarray | None = None,
    ):
        self.case = case
        self.config = config
        self.partition = partition
        self.observation = observation
        self.prior_std = np.asarray(prior_std, dtype=np.float64)
        self.phi0 = partition.phi0
        self.rank = min(int(rank), partition.dimension)
        self.step = step
        self.sigma = sigma
        self.seed = seed
        self.subspace = subspace
        self.factors = factors or FactorCache(None)
        self.exacts = exacts or ExactCache(None)
        self.current = None if current is None else np.asarray(current, dtype=np.float64)
        if self.current is not None and (
            self.current.ndim != 2 or self.current.shape[0] != partition.dimension
        ):
            raise ValueError("current factors must be (dimension, k) in phi coordinates")
        self._runner = None

    @property
    def runner(self) -> ActionRunner:
        """The reused simulator, built on first use so imports stay Warp-free."""
        if self._runner is None:
            self._runner = ActionRunner(self.case, self.config)
        return self._runner

    def run(self, controller, contact) -> Callable[[np.ndarray], np.ndarray]:
        return self.runner.run(
            controller, contact, self.partition, self.observation
        )

    def screen(
        self, actions, progress: Callable[[int, int], None] | None = None
    ) -> list[Entry]:
        """Rank-``rank`` sketch gain of every candidate action."""
        requests = [
            (action.name, self.run(action.controller, action.contact))
            for action in actions
        ]

        def key(index: int, name: str) -> str:
            return factor_key(
                self.case,
                self.config,
                self.phi0,
                actions[index].controller,
                actions[index].contact,
                self.rank,
                self.step,
                self.sigma,
                subspace_seed(self.seed, index, self.subspace),
            )

        return build_dictionary(
            requests,
            self.phi0,
            self.prior_std,
            self.sigma,
            rank=self.rank,
            step=self.step,
            seed=self.seed,
            subspace=self.subspace,
            progress=progress,
            cache=self.factors,
            key=key,
        )

    def fisher(self, controller, contact) -> Fisher:
        """Exact Fisher of one trajectory at ``phi0``."""
        return exact_fisher(
            self.run(controller, contact), self.phi0, self.step, self.sigma, self.prior_std
        )

    def _key(self, controller, contact) -> str:
        return exact_key(
            self.case, self.config, self.phi0, controller, contact, self.step, self.sigma
        )

    def matrix(self, controller, contact) -> np.ndarray:
        """Exact Fisher matrix of a trajectory that was already measured."""
        cached = self.exacts.get(self._key(controller, contact))
        if cached is None:
            raise ValueError("trajectory has not been measured exactly")
        return cached[1]

    def exact(self, name: str, controller, contact, log: Callable[[str], None] = print) -> float:
        """Exact Fisher log-det of one trajectory, cached by its own bytes."""
        key = self._key(controller, contact)
        cached = self.exacts.get(key)
        if cached is None:
            log(f"exact Fisher {name}: {2 * self.partition.dimension + 1} rollouts")
            fisher = self.fisher(controller, contact)
            gain = float(fisher.logdet())
            self.exacts.put(key, gain, fisher.matrix)
            return gain
        return float(cached[0])

    def score(self, controller, contact) -> float:
        """Sketch gain of a candidate against the accumulated factors."""
        sketch = randomized_sketch(
            self.run(controller, contact),
            self.phi0,
            self.prior_std,
            self.sigma,
            self.rank,
            self.step,
            self.seed,
        )
        return gain_against(sketch.U, self.current)

    def decide(
        self, entries: Sequence[Entry], actions, count: int, exact_top: int
    ) -> Decision:
        """Choose from screened entries, measuring the shortlist exactly."""
        by_name = {action.name: action for action in actions}

        def exact_of(name: str) -> float:
            action = by_name[name]
            return self.exact(name, action.controller, action.contact)

        return arbitrate(entries, count, exact_top, exact_of, self.rank)
