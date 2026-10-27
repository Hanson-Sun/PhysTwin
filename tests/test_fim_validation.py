"""Spectrum decay diagnostics."""

import unittest

import numpy as np

from asid.fim.validation import spectrum_summary


class SpectrumTests(unittest.TestCase):
    def test_rank_zero_when_the_contribution_is_concentrated(self):
        summary = spectrum_summary(np.array([1e12, 1.0, 1.0, 1.0]), rank=1)
        self.assertEqual(summary["logdet_rank_90"], 1)
        self.assertGreater(summary["logdet_captured"], 0.9)

    def test_flat_spectrum_requires_every_mode(self):
        summary = spectrum_summary(np.ones(20), rank=10)
        self.assertGreaterEqual(summary["logdet_rank_90"], 18)
        self.assertGreater(summary["logdet_captured"], 0.45)
        self.assertEqual(summary["trace_rank_90"], 18)

    def test_zeros_do_not_divide_by_zero(self):
        summary = spectrum_summary(np.zeros(5), rank=5)
        self.assertEqual(summary["logdet_rank_90"], 1)
        self.assertEqual(summary["trace_rank_90"], 1)


if __name__ == "__main__":
    unittest.main()
