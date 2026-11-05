"""Calibration metrics and post-hoc calibration."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
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


class PlattScaler:
    """Unpenalized logistic calibration with Platt-smoothed binary targets."""

    penalty = None
    solver = "L-BFGS-B"
    max_iter = 2000

    def __init__(self, coefficient: np.ndarray, intercept: float) -> None:
        self.coef_ = np.asarray(coefficient, dtype=float).reshape(1, 1)
        self.intercept_ = np.asarray([intercept], dtype=float)
        self.classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        probability = expit(
            np.asarray(logits, dtype=float).reshape(-1) * self.coef_[0, 0]
            + self.intercept_[0]
        )
        return np.column_stack((1 - probability, probability))


def fit_platt_scaler(
    probabilities: np.ndarray, labels: np.ndarray
) -> PlattScaler:
    """Fit Platt scaling on logit(p), using smoothed targets and no penalty."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=int)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or not len(y):
        raise ValueError("probabilities and labels must be non-empty matching vectors")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Platt calibration requires both label classes")
    positive = y == 1
    smoothed_targets = np.where(
        positive,
        (int(positive.sum()) + 1) / (int(positive.sum()) + 2),
        1 / (int((~positive).sum()) + 2),
    )
    logits = np.log(p / (1 - p))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] * logits + parameters[1]
        probabilities_fit = expit(linear)
        loss = np.sum(
            smoothed_targets * np.logaddexp(0.0, -linear)
            + (1 - smoothed_targets) * np.logaddexp(0.0, linear)
        )
        residual = probabilities_fit - smoothed_targets
        gradient = np.array(
            [np.dot(residual, logits), residual.sum()], dtype=float
        )
        return float(loss), gradient

    fit = minimize(
        objective,
        np.array([1.0, 0.0]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": PlattScaler.max_iter},
    )
    if not fit.success:
        raise RuntimeError(f"Platt calibration failed to converge: {fit.message}")
    return PlattScaler(fit.x[:1], float(fit.x[1]))
