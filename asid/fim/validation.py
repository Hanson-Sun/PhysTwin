"""Spectrum decay diagnostics."""

from __future__ import annotations

import numpy as np


def _capture(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total <= 0.0:
        return np.ones(len(values))
    return np.cumsum(values) / total


def _rank(capture: np.ndarray, fraction: float) -> int:
    return int(min(np.searchsorted(capture, fraction) + 1, len(capture)))


def spectrum_summary(eigenvalues: np.ndarray, rank: int) -> dict[str, int | float]:
    values = np.sort(np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0))[::-1]
    contributions = np.log1p(values)
    logdet_capture = _capture(contributions)
    return {
        "logdet_rank_90": _rank(logdet_capture, 0.90),
        "logdet_rank_95": _rank(logdet_capture, 0.95),
        "logdet_captured": float(
            contributions[:rank].sum() / max(contributions.sum(), 1e-30)
        ),
        "trace_rank_90": _rank(_capture(values), 0.90),
    }
