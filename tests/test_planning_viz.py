"""Planning figures render from arrays."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from asid.planning.actions import Action
from asid.planning.viz import (
    plot_action_gains,
    plot_gain_comparison,
    plot_sequential_scores,
)


def synthetic_actions() -> list[Action]:
    rng = np.random.default_rng(0)
    actions = []
    for index, degrees in enumerate([0, 120, 240]):
        angle = np.deg2rad(degrees)
        start = np.array([0.1 * np.cos(angle), 0.1 * np.sin(angle), 0.3])
        contact = np.repeat(start[None, None], 5, axis=1) + rng.normal(
            scale=0.01, size=(4, 5, 3)
        )
        contact[:, :, 2] -= 0.1 * np.arange(4)[:, None]
        actions.append(
            Action(
                name=f"az{degrees:03d}",
                azimuth=angle,
                controller=contact.copy(),
                contact=contact.astype(np.float32),
            )
        )
    return actions


class PlanningFigureTests(unittest.TestCase):
    def test_action_gain_figure(self):
        nodes = np.random.default_rng(1).normal(scale=0.1, size=(40, 3))
        groups = np.arange(40) % 4
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gains.png"
            plot_action_gains(
                nodes,
                groups,
                synthetic_actions(),
                [1.0, 3.0, 2.0],
                ["az000", "az120"],
                path,
                title="test",
                contact_radius=0.1,
            )
            self.assertGreater(path.stat().st_size, 10_000)

    def test_sequential_figure(self):
        rounds = [
            {
                "round": 1,
                "scores": {"az000": 78.0, "az240": 199.7, "az180": 144.4},
                "selected": "az240",
            },
            {
                "round": 2,
                "scores": {"az000": 22.7, "az180": 28.6, "az120": 26.3},
                "selected": "az180",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequential.png"
            plot_sequential_scores(rounds, path, title="test")
            self.assertGreater(path.stat().st_size, 10_000)

    def test_comparison_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranking.png"
            plot_gain_comparison(
                ["a", "b", "c"], [1.0, 2.0, 3.0], [1.1, 2.1, 2.9], path
            )
            self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
