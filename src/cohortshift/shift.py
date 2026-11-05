"""Unsupervised domain-shift estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import cohort_standardize, within_sample_rank

E0_METHODS = ("none", "A1", "A2", "A3")
A3_WEIGHT_CLIP = (0.1, 10.0)


def _domain_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )


def _require_complete_expression(
    source: pd.DataFrame, target: pd.DataFrame
) -> None:
    if source.isna().any().any() or target.isna().any().any():
        raise ValueError(
            "E0 requires genes filtered to complete measurements in both cohorts"
        )


def _adapt_domain_features(
    source: pd.DataFrame, target: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "none" or method == "A3":
        return source.copy(), target.copy()
    if method == "A1":
        source_scaled, _, _ = cohort_standardize(source)
        target_scaled, _, _ = cohort_standardize(target)
        return source_scaled, target_scaled
    if method == "A2":
        return within_sample_rank(source), within_sample_rank(target)
    raise ValueError(f"Unknown E0 adaptation method: {method}")


def _source_density_weights(
    train_features: np.ndarray, train_domains: np.ndarray
) -> np.ndarray:
    source_mask = train_domains == 0
    source_count = int(source_mask.sum())
    target_count = len(train_domains) - source_count
    if not source_count or not target_count:
        raise ValueError("each domain-classifier training fold needs both cohorts")
    estimator = _domain_classifier().fit(train_features, train_domains)
    probability = estimator.predict_proba(train_features[source_mask])[:, 1]
    ratio = probability / np.maximum(1 - probability, 1e-8)
    ratio *= source_count / target_count
    weights = np.ones(len(train_domains), dtype=float)
    weights[source_mask] = np.clip(
        ratio, A3_WEIGHT_CLIP[0], A3_WEIGHT_CLIP[1]
    )
    return weights


def domain_classifier_auroc(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    method: str = "none",
    seed: int = 0,
    folds: int = 5,
) -> float:
    """Unweighted cross-validated source-vs-target AUROC."""
    _require_complete_expression(source, target)
    source, target = _adapt_domain_features(source, target, method)
    features = np.vstack((source.to_numpy(dtype=float), target.to_numpy(dtype=float)))
    domains = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
    fold_count = min(folds, int(np.bincount(domains).min()))
    if fold_count < 2:
        raise ValueError("at least two samples per cohort are required for CV AUROC")
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_indices, test_indices in splitter.split(features, domains):
        classifier = _domain_classifier()
        classifier.fit(features[train_indices], domains[train_indices])
        target_scores = classifier.predict_proba(features[test_indices])[:, 1]
        aucs.append(float(roc_auc_score(domains[test_indices], target_scores)))
    return float(np.mean(aucs))


def _weighted_ks(
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_weights: np.ndarray,
) -> float:
    values = np.sort(np.concatenate((source_values, target_values)))
    source_order = np.argsort(source_values)
    ordered_source = source_values[source_order]
    ordered_weights = source_weights[source_order]
    source_cdf = np.searchsorted(
        ordered_source, values, side="right"
    )
    source_cdf = np.array(
        [
            ordered_weights[:count].sum() / ordered_weights.sum()
            for count in source_cdf
        ]
    )
    target_ordered = np.sort(target_values)
    target_cdf = np.searchsorted(target_ordered, values, side="right") / len(
        target_ordered
    )
    return float(np.max(np.abs(source_cdf - target_cdf)))


def per_gene_ks(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    method: str = "none",
) -> pd.DataFrame:
    """Return one two-sample KS statistic per gene after the requested transform."""
    source, target = _adapt_domain_features(source, target, method)
    source_weights = pd.Series(1.0, index=source.index)
    if method == "A3":
        _require_complete_expression(source, target)
        combined = np.vstack((source.to_numpy(), target.to_numpy()))
        domain = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
        source_weights = pd.Series(
            _source_density_weights(combined, domain)[: len(source)],
            index=source.index,
        )
    rows = []
    for gene in source.columns:
        source_mask = source[gene].notna()
        target_mask = target[gene].notna()
        source_values = source.loc[source_mask, gene].to_numpy(dtype=float)
        target_values = target.loc[target_mask, gene].to_numpy(dtype=float)
        result_row = {
            "gene": gene,
            "n_source": len(source_values),
            "n_target": len(target_values),
            "ks_statistic": np.nan,
            "p_value": np.nan,
        }
        if len(source_values) and len(target_values) and method == "A3":
            result_row["ks_statistic"] = _weighted_ks(
                source_values,
                target_values,
                source_weights.loc[source_mask].to_numpy(),
            )
        elif len(source_values) and len(target_values):
            result = ks_2samp(source_values, target_values, method="auto")
            result_row["ks_statistic"] = float(result.statistic)
            result_row["p_value"] = float(result.pvalue)
        rows.append(result_row)
    return pd.DataFrame(rows)


def save_pca_overlap_plot(
    source: pd.DataFrame,
    target: pd.DataFrame,
    output_path: str,
    *,
    method: str = "none",
    seed: int = 0,
) -> None:
    """Save an unsupervised PCA overlap plot for the requested adaptation."""
    _require_complete_expression(source, target)
    source, target = _adapt_domain_features(source, target, method)
    values = np.vstack((source.to_numpy(dtype=float), target.to_numpy(dtype=float)))
    scaled = StandardScaler().fit_transform(values)
    coordinates = PCA(
        n_components=2, svd_solver="randomized", random_state=seed
    ).fit_transform(scaled)
    source_count = len(source)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        coordinates[:source_count, 0],
        coordinates[:source_count, 1],
        s=18,
        alpha=0.55,
        label="Beat AML",
    )
    axis.scatter(
        coordinates[source_count:, 0],
        coordinates[source_count:, 1],
        s=24,
        alpha=0.65,
        label="TCGA-LAML",
    )
    axis.set(
        xlabel="PC1",
        ylabel="PC2",
        title=f"E0 PCA overlap ({method})",
    )
    axis.legend()
    figure.tight_layout()
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def run_e0_shift(
    source: pd.DataFrame,
    target: pd.DataFrame,
    methods: tuple[str, ...] | list[str] = E0_METHODS,
    *,
    seed: int = 0,
    pca_output_dir: str = "results/figures",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute E0 domain AUROC, PCA, and per-gene KS for each method."""
    if not methods:
        raise ValueError("at least one E0 method is required")
    if len(set(methods)) != len(methods):
        raise ValueError("E0 methods must not contain duplicates")
    _require_complete_expression(source, target)
    summary_rows = []
    ks_frames = []
    for method in methods:
        gene_ks = per_gene_ks(source, target, method=method)
        gene_ks.insert(0, "method", method)
        ks_frames.append(gene_ks)
        save_pca_overlap_plot(
            source,
            target,
            f"{pca_output_dir}/e0_pca_{method}.png",
            method=method,
            seed=seed,
        )
        stats = gene_ks["ks_statistic"]
        summary_rows.append(
            {
                "method": method,
                "expression_units": (
                    "Beat AML: log2 RPKM; TCGA-LAML: batch-corrected RSEM"
                ),
                "source_missing_cells": int(source.isna().sum().sum()),
                "target_missing_cells": int(target.isna().sum().sum()),
                "domain_classifier_auroc_cv": domain_classifier_auroc(
                    source,
                    target,
                    method=method,
                    seed=seed,
                ),
                "domain_classifier_auroc_note": (
                    "unweighted; unchanged by construction (A3 weights are used "
                    "for weighted KS only)"
                    if method == "A3"
                    else "unweighted"
                ),
                "n_genes": len(gene_ks),
                "n_genes_ks_tested": int(stats.notna().sum()),
                "mean_gene_ks": float(stats.mean()),
                "median_gene_ks": float(stats.median()),
                "max_gene_ks": float(stats.max()) if stats.notna().any() else np.nan,
            }
        )
        if method == "A3":
            a3_weights = domain_classifier_weights(
                source.to_numpy(), target.to_numpy(), clip=A3_WEIGHT_CLIP
            )
            effective_sample_size = float(
                np.square(a3_weights.sum()) / np.square(a3_weights).sum()
            )
            max_weight = float(a3_weights.max())
            max_weight_at_upper_clip = bool(
                np.isclose(max_weight, A3_WEIGHT_CLIP[1])
            )
            summary_rows[-1].update(
                {
                    "effective_sample_size": effective_sample_size,
                    "max_weight": max_weight,
                    "clip_lower_bound": A3_WEIGHT_CLIP[0],
                    "clip_upper_bound": A3_WEIGHT_CLIP[1],
                    "max_weight_at_upper_clip": max_weight_at_upper_clip,
                    "degenerate_flag": bool(
                        effective_sample_size < 0.1 * len(source)
                        or max_weight_at_upper_clip
                    ),
                }
            )
        else:
            summary_rows[-1].update(
                {
                    "effective_sample_size": np.nan,
                    "max_weight": np.nan,
                    "clip_lower_bound": np.nan,
                    "clip_upper_bound": np.nan,
                    "max_weight_at_upper_clip": None,
                    "degenerate_flag": None,
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(ks_frames, ignore_index=True)


def domain_classifier_weights(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    clip: tuple[float, float] = A3_WEIGHT_CLIP,
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
