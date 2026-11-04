"""Small orchestration helpers for pre-registered method evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .adapt import adaptation_weights, adapt_features
from .calibration import calibration_metrics
from .models import logistic_model

METHODS = ("none", "A1", "A2", "A3")


def evaluate_source_model(
    source_features: pd.DataFrame,
    source_labels: np.ndarray,
    target_features: pd.DataFrame,
    target_labels: np.ndarray,
    *,
    method: str = "none",
) -> dict[str, float]:
    """Fit on source outcomes and evaluate using target outcomes only here."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    source_x, target_x = adapt_features(
        method if method in {"A1", "A2"} else "none",
        source_features,
        target_features,
    )
    model = logistic_model()
    fit_kwargs = {}
    if method == "A3":
        fit_kwargs["sample_weight"] = adaptation_weights(
            source_x.to_numpy(), target_x.to_numpy()
        )
    model.fit(source_x, source_labels, **fit_kwargs)
    probabilities = model.predict_proba(target_x)[:, 1]
    return calibration_metrics(target_labels, probabilities)


def run_methods(
    source_features: pd.DataFrame,
    source_labels: np.ndarray,
    target_features: pd.DataFrame,
    target_labels: np.ndarray,
    *,
    evaluator: Callable[..., dict[str, float]] = evaluate_source_model,
) -> dict[str, dict[str, float]]:
    """Evaluate the fixed method list without data-driven target tuning."""
    return {
        method: evaluator(
            source_features,
            source_labels,
            target_features,
            target_labels,
            method=method,
        )
        for method in METHODS
    }
