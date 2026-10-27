"""Overlay rendering."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from asid.fim.viz import (
    plot_fim_matrix,
    plot_fim_overlay,
    plot_point_information,
    point_information,
)


class InformationTests(unittest.TestCase):
    def test_per_node_gain_matches_gaussian_logdet(self):
        rng = np.random.default_rng(3)
        jac = rng.normal(size=(4, 3 * 5 * 3))
        std = np.array([0.2, 0.3, 0.4, 0.5])
        sigma = 0.1
        values = point_information(jac, std, sigma, nodes=5)
        weighted = (jac * std[:, None] / sigma).reshape(4, 3, 5, 3)
        expected = []
        for node in range(5):
            block = weighted[:, :, node, :].reshape(4, -1)
            expected.append(0.5 * np.linalg.slogdet(np.eye(4) + block @ block.T)[1])
        np.testing.assert_allclose(values, expected)
        self.assertTrue(np.all(values >= 0))

    def test_matrix_and_per_node_plots_are_written(self):
        nodes = np.random.default_rng(4).normal(size=(8, 3))
        with tempfile.TemporaryDirectory() as tmp:
            matrix = plot_fim_matrix(np.eye(3), Path(tmp) / "matrix.png")
            points = plot_point_information(nodes, np.arange(8), Path(tmp) / "points.png")
            self.assertGreater(matrix.stat().st_size, 1000)
            self.assertGreater(points.stat().st_size, 1000)


class OverlayTests(unittest.TestCase):
    def test_writes_a_figure_for_both_parameter_types(self):
        rng = np.random.default_rng(0)
        nodes = rng.normal(size=(40, 3))
        midpoints = rng.normal(size=(60, 3))
        node_groups = rng.integers(0, 4, size=40)
        spring_groups = rng.integers(0, 4, size=60)
        diag = np.abs(rng.normal(size=9)) * 1e4
        indices = {"k": slice(0, 4), "c": slice(4, 5), "m": slice(5, 9)}
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_fim_overlay(
                nodes,
                midpoints,
                node_groups,
                spring_groups,
                diag,
                indices,
                Path(tmp) / "figs" / "overlay.png",
                title="test",
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
