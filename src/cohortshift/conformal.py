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
    if p.ndim == 1:
        p = np.column_stack((1 - p, p))
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("probabilities must be a binary vector or two-column matrix")
    return [
        tuple(label for label, score in enumerate((1 - row[0], 1 - row[1])) if score <= threshold)
        for row in p
    ]


def conformal_coverage_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Summarize split-conformal set coverage overall and by true class."""
    y = np.asarray(labels, dtype=int)
    sets = prediction_sets(probabilities, threshold)
    if y.ndim != 1 or len(y) != len(sets) or not len(y):
        raise ValueError("labels and probabilities must be non-empty matching rows")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("conformal labels must be binary")
    covered = np.array([label in prediction_set for label, prediction_set in zip(y, sets)])
    positive = y == 1
    negative = y == 0
    return {
        "coverage": float(covered.mean()),
        "positive_class_coverage": (
            float(covered[positive].mean()) if positive.any() else float("nan")
        ),
        "negative_class_coverage": (
            float(covered[negative].mean()) if negative.any() else float("nan")
        ),
        "mean_prediction_set_size": float(np.mean([len(value) for value in sets])),
        "empty_set_fraction": float(np.mean([not value for value in sets])),
    }
