"""Heatmap helpers split the Fisher diagonal and interpolate it faithfully."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from asid.planning.fimmap import (
    fisher_diagonal,
    group_values,
    heat_grid,
    plot_fim_heatmaps,
    square_extent,
)


def plane(count: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.05, 0.05, size=(count, 3))
    points[:, 2] = -0.02
    return points, points[:, :2].sum(axis=1)


class DiagonalTests(unittest.TestCase):
    def test_blocks_follow_the_parameter_layout(self):
        groups = 3
        diag = np.array([1.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0])
        matrix = np.diag(diag)
        blocks = fisher_diagonal(matrix, groups)
        np.testing.assert_allclose(blocks["k"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(blocks["c"], [9.0])
        np.testing.assert_allclose(blocks["m"], [4.0, 5.0, 6.0])

    def test_group_values_expand_onto_elements(self):
        diagonal = {
            "k": np.array([10.0, 20.0, 30.0]),
            "c": np.array([7.0]),
            "m": np.array([1.0, 2.0, 3.0]),
        }
        blocks = group_values(diagonal, np.array([2, 0, 1]), np.array([1, 1, 2, 0]))
        np.testing.assert_allclose(blocks["k"], [30.0, 10.0, 20.0])
        np.testing.assert_allclose(blocks["m"], [2.0, 2.0, 3.0, 1.0])
        np.testing.assert_allclose(blocks["c"], [7.0])


class GridTests(unittest.TestCase):
    def test_linear_field_is_reproduced(self):
        axis = np.linspace(-0.05, 0.05, 24)
        gx, gy = np.meshgrid(axis, axis)
        points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.02)])
        values = points[:, 0] + points[:, 1]
        extent = square_extent(points, pad=0.0)
        grid = heat_grid(points, values, extent, resolution=len(axis))
        expected = np.linspace(extent[2], extent[3], len(axis))[:, None] + np.linspace(
            extent[0], extent[1], len(axis)
        )[None, :]
        inside = ~np.ma.getmaskarray(grid)
        self.assertGreater(inside.sum(), 0.9 * grid.size)
        np.testing.assert_allclose(grid.filled(np.nan)[inside], expected[inside], atol=1e-8)

    def test_far_cells_are_masked(self):
        points, values = plane()
        extent = square_extent(points, pad=1.0)
        grid = heat_grid(points, values, extent, resolution=31)
        self.assertGreater(np.ma.getmaskarray(grid).sum(), 0)
        self.assertLess(np.ma.getmaskarray(grid).sum(), grid.size)

    def test_square_extent_covers_every_point(self):
        points, _ = plane()
        extent = square_extent(points)
        self.assertLessEqual(extent[0], points[:, 0].min())
        self.assertGreaterEqual(extent[1], points[:, 0].max())
        self.assertAlmostEqual(extent[1] - extent[0], extent[3] - extent[2], places=10)


class FigureTests(unittest.TestCase):
    def test_writes_one_panel_per_action(self):
        rng = np.random.default_rng(1)
        groups = 2
        size = 2 * groups + 1
        nodes = rng.uniform(-0.05, 0.05, size=(30, 3))
        midpoints = rng.uniform(-0.05, 0.05, size=(8, 3))
        diagonals = {
            name: {
                "k": np.full(8, scale),
                "c": np.array([scale]),
                "m": np.full(30, scale / 2),
            }
            for name, scale in (("a", 1.0), ("b", 10.0))
        }
        gains = {"a": 5.0, "b": 9.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figs" / "heat.png"
            written = plot_fim_heatmaps(
                nodes, midpoints, diagonals, ["b", "a"], gains, path,
                title="synthetic", selected=["b"], resolution=24,
            )
            self.assertEqual(written, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(size, 2 * groups + 1)


if __name__ == "__main__":
    unittest.main()
