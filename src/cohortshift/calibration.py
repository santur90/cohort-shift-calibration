"""Calibration metrics and post-hoc calibration."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def calibration_intercept_slope(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Estimate logistic calibration intercept and slope from probabilities."""
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Compute Brier score, calibration intercept/slope, and AUROC."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "auroc": float(roc_auc_score(y, p)),
    }


def fit_platt_scaler(
    probabilities: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    """Fit a Platt scaler using calibration/source data only."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(
        logits, np.asarray(labels, dtype=int)
    )
