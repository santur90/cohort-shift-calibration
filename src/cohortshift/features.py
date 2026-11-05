"""Expression transformations used in the registered method comparison."""

from __future__ import annotations

import pandas as pd


def select_top_variance_genes(
    source_train: pd.DataFrame, n_genes: int = 2000
) -> list[str]:
    """Rank genes by variance using only the supplied source-training samples."""
    if n_genes < 1:
        raise ValueError("n_genes must be positive")
    variances = source_train.var(axis=0, ddof=0).replace([float("inf")], float("nan"))
    variances = variances.fillna(float("-inf"))
    ranked = variances.sort_values(ascending=False, kind="mergesort")
    return ranked.index[: min(n_genes, len(ranked))].tolist()


def cohort_standardize(
    values: pd.DataFrame, *, mean: pd.Series | None = None, scale: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize each gene using the provided cohort's feature distribution."""
    fitted_mean = values.mean(axis=0) if mean is None else mean
    fitted_scale = values.std(axis=0, ddof=0) if scale is None else scale
    fitted_scale = fitted_scale.mask(fitted_scale == 0, 1.0)
    return (values - fitted_mean) / fitted_scale, fitted_mean, fitted_scale


def within_sample_rank(values: pd.DataFrame) -> pd.DataFrame:
    """Convert each sample row to fractional within-sample gene ranks."""
    return values.rank(axis=1, method="average", pct=True)
