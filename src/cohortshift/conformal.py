"""Split-conformal classification utilities."""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Return the finite-sample corrected upper quantile of nonconformity."""
    values = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    rank = min(int(np.ceil((values.size + 1) * (1 - alpha))), values.size)
    return float(np.sort(values)[rank - 1])


def prediction_sets(
    probabilities: np.ndarray, threshold: float
) -> list[tuple[int, ...]]:
    """Build class sets from class-conditional 1-p probabilities."""
    p = np.asarray(probabilities, dtype=float)
    return [
        tuple(label for label, score in enumerate((1 - row[0], 1 - row[1])) if score <= threshold)
        for row in p
    ]
