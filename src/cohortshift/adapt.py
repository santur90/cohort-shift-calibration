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
    *,
    source_train_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a registered, unsupervised transformation to source and target X."""
    if list(source_features.columns) != list(target_features.columns):
        raise ValueError("source and target features must have identical ordered columns")
    if source_train_features is not None and list(source_train_features.columns) != list(
        source_features.columns
    ):
        raise ValueError("source training features must have identical ordered columns")
    if method in {"none", "raw", "A3", "A0+A3"}:
        target = target_features.copy()
        if method in {"A0+A3"}:
            target = _target_a0(target)
        return source_features.copy(), target
    if method == "A0":
        return source_features.copy(), _target_a0(target_features)
    if method in {"A1", "A0+A1"}:
        source_train = (
            source_features if source_train_features is None else source_train_features
        )
        target_values = (
            _target_a0(target_features) if method == "A0+A1" else target_features
        )
        _, mean, scale = cohort_standardize(source_train)
        source, _, _ = cohort_standardize(
            source_features, mean=mean, scale=scale
        )
        target, _, _ = cohort_standardize(target_values)
        return source, target
    if method == "A2":
        return within_sample_rank(source_features), within_sample_rank(target_features)
    raise ValueError(f"Unknown feature adaptation method: {method}")


def _target_a0(values: pd.DataFrame) -> pd.DataFrame:
    transformed = np.log2(np.maximum(values.to_numpy(dtype=float), 0.0) + 1.0)
    return pd.DataFrame(transformed, index=values.index, columns=values.columns)


def adaptation_weights(
    source_features: np.ndarray, target_features: np.ndarray
) -> np.ndarray:
    """Return A3 clipped domain-classifier weights from features alone."""
    return domain_classifier_weights(source_features, target_features, clip=(0.1, 10.0))
