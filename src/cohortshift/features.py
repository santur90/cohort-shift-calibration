"""Expression transformations used in the registered method comparison."""

from __future__ import annotations

import pandas as pd


def filter_jointly_complete_genes(
    source: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep shared genes measured in every sample of both cohorts."""
    source_genes = set(source.columns)
    target_genes = set(target.columns)
    shared_genes = source_genes & target_genes
    complete_genes = {
        gene
        for gene in shared_genes
        if source[gene].notna().all() and target[gene].notna().all()
    }
    ordered_genes = sorted(complete_genes)
    dropped_source = len(
        (source_genes - shared_genes)
        | {gene for gene in shared_genes if not source[gene].notna().all()}
    )
    dropped_target = len(
        (target_genes - shared_genes)
        | {gene for gene in shared_genes if not target[gene].notna().all()}
    )
    dropped_total = len((source_genes | target_genes) - complete_genes)
    return (
        source.loc[:, ordered_genes].copy(),
        target.loc[:, ordered_genes].copy(),
        {
            "source": dropped_source,
            "target": dropped_target,
            "total": dropped_total,
        },
    )


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
