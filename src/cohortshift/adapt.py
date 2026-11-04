"""Feature-only adaptation interfaces; no target-label arguments are accepted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import cohort_standardize, within_sample_rank
from .shift import domain_classifier_weights


def adapt_features(
    method: str,
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a registered, unsupervised transformation to source and target X."""
    if list(source_features.columns) != list(target_features.columns):
        raise ValueError("source and target features must have identical ordered columns")
    if method == "none":
        return source_features.copy(), target_features.copy()
    if method == "A1":
        source, mean, scale = cohort_standardize(source_features)
        target, _, _ = cohort_standardize(target_features)
        del mean, scale
        return source, target
    if method == "A2":
        return within_sample_rank(source_features), within_sample_rank(target_features)
    raise ValueError(f"Unknown feature adaptation method: {method}")


def adaptation_weights(
    source_features: np.ndarray, target_features: np.ndarray
) -> np.ndarray:
    """Return A3 clipped domain-classifier weights from features alone."""
    return domain_classifier_weights(source_features, target_features, clip=(0.1, 10.0))
