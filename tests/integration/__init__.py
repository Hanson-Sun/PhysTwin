"""Integration suite: ``python -m unittest tests.integration``.

Full-pipeline tests (real-asset meshing, long rollouts) that take minutes
rather than seconds. The quick smoke suite is ``python -m unittest tests``.
"""

import unittest

INTEGRATION_MODULES = (
    "tests.integration.test_soft_rollouts",
    "tests.integration.test_soft_sloth",
)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in INTEGRATION_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    return suite
