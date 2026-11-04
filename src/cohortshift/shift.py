"""Unsupervised domain-shift estimation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def domain_classifier_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    clip: tuple[float, float] = (0.1, 10.0),
) -> np.ndarray:
    """Estimate clipped target/source density ratios without target outcomes."""
    source = np.asarray(source_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target features must be 2D with matching columns")
    if not 0 < clip[0] <= clip[1]:
        raise ValueError("clip bounds must satisfy 0 < lower <= upper")
    combined = np.vstack((source, target))
    domain = np.r_[np.zeros(len(source)), np.ones(len(target))]
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
    ).fit(combined, domain)
    target_probability = classifier.predict_proba(source)[:, 1]
    prior_ratio = len(source) / max(len(target), 1)
    weights = target_probability / np.maximum(1 - target_probability, 1e-8) * prior_ratio
    return np.clip(weights, clip[0], clip[1])
