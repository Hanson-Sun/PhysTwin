"""Quick smoke suite: ``python -m unittest tests``.

Everything here verifies core functionality and finishes in about a minute.
Heavy full-pipeline tests live in ``tests.integration`` and run with
``python -m unittest tests.integration``.
"""

import unittest

SMOKE_MODULES = (
    "tests.test_cuda_arch",
    "tests.test_eigh_3x3",
    "tests.test_cuda_linalg_backend",
    "tests.test_strided_render_compositing",
    "tests.test_runtime_visualization_cli",
    "tests.test_cuda13_runtime_hooks",
    "tests.test_controller_proximity",
    "tests.test_soft_body",
    "tests.test_sim_export",
    "tests.test_interior_sample",
)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in SMOKE_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite
