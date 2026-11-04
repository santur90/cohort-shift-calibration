"""Expression transformations used in the registered method comparison."""

from __future__ import annotations

import pandas as pd


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
