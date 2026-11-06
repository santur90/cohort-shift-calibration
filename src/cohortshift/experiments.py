"""Small orchestration helpers for pre-registered method evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adapt import adaptation_weights, adapt_features
from .conformal import conformal_coverage_metrics, conformal_quantile
from .features import select_top_variance_genes
from .shift import domain_classifier_auroc, per_gene_ks
from .source_experiments import (
    _apply_calibrator,
    _fit_calibrator,
    _metric_values,
    _tuned_estimators,
)

E3_METHODS = ("raw", "A0", "A0+A1", "A2", "A3", "A0+A3")
E3_CALIBRATORS = ("none", "Platt", "isotonic")
E3_MODELS = ("logistic", "hist_gradient_boosting")
E3_BOOTSTRAP_RESAMPLES = 2000


def predict_adapted_probabilities(
    source_train: pd.DataFrame,
    source_train_labels: np.ndarray,
    source_calibration: pd.DataFrame,
    source_calibration_labels: np.ndarray,
    source_test: pd.DataFrame,
    target_features: pd.DataFrame,
    *,
    method: str,
    seed: int,
) -> dict[str, object]:
    """Train/calibrate using source outcomes and predict from target features only."""
    if method not in E3_METHODS:
        raise ValueError("unknown registered E3 method")
    if (
        list(source_train.columns) != list(source_calibration.columns)
        or list(source_train.columns) != list(source_test.columns)
        or list(source_train.columns) != list(target_features.columns)
    ):
        raise ValueError("all source and target matrices need identical gene columns")
    source_labels = np.asarray(source_train_labels, dtype=int)
    calibration_labels = np.asarray(source_calibration_labels, dtype=int)
    train_x, target_x = adapt_features(
        method,
        source_train,
        target_features,
        source_train_features=source_train,
    )
    calibration_x, _ = adapt_features(
        method,
        source_calibration,
        target_features,
        source_train_features=source_train,
    )
    test_x, _ = adapt_features(
        method,
        source_test,
        target_features,
        source_train_features=source_train,
    )
    weights = None
    weight_diagnostics: dict[str, float | bool | str] = {}
    if method in {"A3", "A0+A3"}:
        weights = adaptation_weights(train_x.to_numpy(), target_x.to_numpy())
        effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())
        lower_fraction = float(np.isclose(weights, 0.1).mean())
        upper_fraction = float(np.isclose(weights, 10.0).mean())
        reasons = []
        if effective_sample_size < 0.1 * len(weights):
            reasons.append("ESS below 10% of source N")
        if np.isclose(weights.max(), 10.0):
            reasons.append("maximum weight at upper clip")
        if lower_fraction >= 0.5:
            reasons.append("at least 50% of weights at lower clip")
        if upper_fraction >= 0.5:
            reasons.append("at least 50% of weights at upper clip")
        if np.allclose(weights, weights[0]):
            reasons.append("all weights equal")
        weight_diagnostics = {
            "effective_sample_size": effective_sample_size,
            "weight_min": float(weights.min()),
            "weight_median": float(np.median(weights)),
            "max_weight": float(weights.max()),
            "frac_weights_at_lower_clip": lower_fraction,
            "frac_weights_at_upper_clip": upper_fraction,
            "clip_lower_bound": 0.1,
            "clip_upper_bound": 10.0,
            "degenerate_flag": bool(reasons),
            "degenerate_note": "; ".join(reasons) if reasons else "No degeneracy criteria met",
            "status": "failed" if reasons else "ready",
            "failure_reason": (
                "A3 degenerate; registered classifier and clip were not changed"
                if reasons
                else ""
            ),
        }
        if reasons:
            return {
                "failed": True,
                "status": "failed",
                "failure_reason": weight_diagnostics["failure_reason"],
                "weight_diagnostics": weight_diagnostics,
            }
        weights = weights / weights.mean()
    estimators = _tuned_estimators(
        train_x,
        source_labels,
        seed=seed,
        scale_logistic=method not in {"A1", "A0+A1"},
        sample_weights=weights,
    )
    predictions: dict[str, object] = {}
    for name, (estimator, parameters) in estimators.items():
        calibration_probability = estimator.predict_proba(calibration_x)[:, 1]
        test_probability = estimator.predict_proba(test_x)[:, 1]
        target_probability = estimator.predict_proba(target_x)[:, 1]
        calibrators = {
            kind: _fit_calibrator(kind, calibration_probability, calibration_labels)
            for kind in E3_CALIBRATORS
        }
        predictions[name] = {
            "hyperparameters": parameters,
            "estimator": estimator,
            "calibrators": calibrators,
            "source_calibration_probabilities": calibration_probability,
            "source_test_probabilities": test_probability,
            "target_probabilities": target_probability,
        }
    return {
        "failed": False,
        "status": "ready",
        "predictions": predictions,
        "weight_diagnostics": weight_diagnostics,
        "adapted_target_features": target_x,
    }


def prepare_cross_cohort_predictions(
    source_expression: pd.DataFrame,
    source_labels: pd.Series,
    target_expression: pd.DataFrame,
    registered_main_split: dict[str, str],
    repeated_splits: pd.DataFrame,
    *,
    registered_main_genes: list[str] | None = None,
    repeat_genes: dict[int, list[str]] | None = None,
    seed: int = 3101,
    outcome: str = "NPM1",
    methods: tuple[str, ...] = E3_METHODS,
    include_repeats: bool = True,
) -> dict[str, object]:
    """Prepare all feature-only E3/E4 predictions without loading target labels."""
    if outcome not in {"NPM1", "DNMT3A"}:
        raise ValueError("outcome must be NPM1 or DNMT3A")
    if source_expression.index.has_duplicates or target_expression.index.has_duplicates:
        raise ValueError("source and target sample IDs must be unique")
    if set(source_expression.index) != set(source_labels.index):
        raise ValueError("source expression and labels must have identical sample IDs")
    if set(source_expression.index) != set(registered_main_split):
        raise ValueError("registered main split must match source sample IDs")
    if set(registered_main_split.values()) != {"train", "validation", "test"}:
        raise ValueError("registered main split must include train/validation/test")
    if not {"repeat", "sample_id", "split", "split_set"}.issubset(
        repeated_splits.columns
    ):
        raise ValueError("saved repeated split table has an invalid schema")
    if set(target_expression.columns) != set(source_expression.columns):
        raise ValueError("source and target expression must use the same gene universe")
    target_expression = target_expression.loc[:, source_expression.columns]
    source_labels = source_labels.loc[source_expression.index]
    if source_labels.isna().any() or not set(source_labels.unique()).issubset({0, 1}):
        raise ValueError("source NPM1 labels must be fully profiled binary values")
    if registered_main_genes is None:
        main_train = [
            sample
            for sample, split in registered_main_split.items()
            if split == "train"
        ]
        registered_main_genes = select_top_variance_genes(
            source_expression.loc[main_train], n_genes=2000
        )
    if not registered_main_genes or not set(registered_main_genes).issubset(
        source_expression.columns
    ):
        raise ValueError("registered main gene list is empty or not in aligned data")
    def run_split(
        split_values: dict[str, str],
        *,
        split_set: str,
        repeat: int | str,
        split_genes: list[str] | None,
        output_entries: list[dict[str, object]],
    ) -> None:
        train_ids = [
            sample for sample, split in split_values.items() if split == "train"
        ]
        calibration_ids = [
            sample
            for sample, split in split_values.items()
            if split in {"calibration", "validation"}
        ]
        test_ids = [
            sample for sample, split in split_values.items() if split == "test"
        ]
        if not train_ids or not calibration_ids or not test_ids:
            raise ValueError(f"{split_set} must have non-empty train/calibration/test")
        train_x_all = source_expression.loc[train_ids]
        genes = (
            select_top_variance_genes(train_x_all, n_genes=2000)
            if split_genes is None
            else split_genes
        )
        if not genes or len(genes) != len(set(genes)):
            raise ValueError(f"{split_set} gene list must be non-empty and unique")
        if not set(genes).issubset(source_expression.columns):
            raise ValueError(f"{split_set} gene list contains unknown symbols")
        train_x = source_expression.loc[train_ids, genes]
        calibration_x = source_expression.loc[calibration_ids, genes]
        test_x = source_expression.loc[test_ids, genes]
        target_x = target_expression.loc[:, genes]
        source_all_x = source_expression.loc[:, genes]
        domain_diagnostics: dict[str, dict[str, float | str]] = {}
        for method in methods:
            domain_method = "none" if method == "raw" else method
            domain_auc = domain_classifier_auroc(
                source_all_x, target_x, method=domain_method, seed=seed
            )
            gene_ks = per_gene_ks(
                source_all_x, target_x, method=domain_method
            )
            domain_diagnostics[method] = {
                "domain_classifier_auroc_cv": float(domain_auc),
                "mean_gene_ks": float(gene_ks["ks_statistic"].mean()),
                "domain_classifier_auroc_note": (
                    "unweighted; unchanged by construction (A3 weights are used "
                    "for model training)"
                    if method in {"A3", "A0+A3"}
                    else "unweighted"
                ),
            }
        train_y = source_labels.loc[train_ids].to_numpy(dtype=int)
        calibration_y = source_labels.loc[calibration_ids].to_numpy(dtype=int)
        test_y = source_labels.loc[test_ids].to_numpy(dtype=int)
        for method in methods:
            result = predict_adapted_probabilities(
                train_x,
                train_y,
                calibration_x,
                calibration_y,
                test_x,
                target_x,
                method=method,
                seed=seed + int(repeat) if isinstance(repeat, int) else seed,
            )
            diagnostics = dict(domain_diagnostics[method])
            diagnostics.update(result.get("weight_diagnostics", {}))
            output_entries.append(
                {
                    "split_set": split_set,
                    "repeat": repeat,
                    "method": method,
                    "outcome": outcome,
                    "failed": bool(result["failed"]),
                    "status": result["status"],
                    "failure_reason": result.get("failure_reason", ""),
                    "diagnostics": diagnostics,
                    "predictions": result.get("predictions", {}),
                    "source_calibration_labels": calibration_y,
                    "source_test_labels": test_y,
                }
            )

    main_entries: list[dict[str, object]] = []
    run_split(
        registered_main_split,
        split_set="registered_main_split",
        repeat="registered_main_split",
        split_genes=registered_main_genes,
        output_entries=main_entries,
    )
    repeat_entries: list[dict[str, object]] = []
    repeats = repeated_splits.loc[repeated_splits["split_set"] == "20_repeats"]
    if not include_repeats:
        repeats = repeats.iloc[0:0]
    elif repeats.empty:
        raise ValueError("saved repeated split table has no 20-repeat assignments")
    for repeat, repeat_rows in repeats.groupby("repeat", sort=True):
        values = repeat_rows.set_index("sample_id")["split"].astype(str).to_dict()
        if set(values) != set(source_expression.index):
            raise ValueError(f"saved repeat {repeat} does not cover source samples")
        if repeat_genes is not None and int(repeat) not in repeat_genes:
            raise ValueError(f"saved gene selection is missing repeat {repeat}")
        gene_list = None if repeat_genes is None else repeat_genes[int(repeat)]
        run_split(
            values,
            split_set="20_repeats",
            repeat=int(repeat),
            split_genes=gene_list,
            output_entries=repeat_entries,
        )
    return {
        "outcome": outcome,
        "target_samples": target_expression.index.tolist(),
        "main": main_entries,
        "repeats": repeat_entries,
    }


def _prediction_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | int | str]:
    metrics, _ = _metric_values(labels, probabilities, include_calibration=True)
    return metrics


def _bootstrap_metric_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: list[str],
    *,
    seed: int,
    n_resamples: int = E3_BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    patients = np.asarray(patient_ids, dtype=str)
    if len(y) != len(p) or len(y) != len(patients):
        raise ValueError("bootstrap inputs must have equal lengths")
    unique_patients, patient_index = np.unique(patients, return_inverse=True)
    if len(unique_patients) != len(y):
        raise ValueError("target bootstrap requires one sample per patient")
    rng = np.random.default_rng(seed)
    metric_names = (
        "auroc",
        "auprc",
        "brier",
        "ece",
        "calibration_intercept",
        "calibration_slope",
    )
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    slope_failure_or_gt5 = 0
    for _ in range(n_resamples):
        sampled_patients = rng.integers(0, len(unique_patients), len(unique_patients))
        indices = np.concatenate(
            [np.flatnonzero(patient_index == index) for index in sampled_patients]
        )
        if not len(indices) or len(np.unique(y[indices])) != 2:
            slope_failure_or_gt5 += 1
            continue
        try:
            metrics = _prediction_metrics(y[indices], p[indices])
        except (ValueError, FloatingPointError):
            slope_failure_or_gt5 += 1
            continue
        slope_failure_or_gt5 += int(
            not bool(metrics["calibration_slope_fit_converged"])
            or float(metrics["calibration_slope"]) > 5
        )
        if float(metrics["frac_pred_at_clip_bounds"]) > 0.05:
            metrics["calibration_slope"] = np.nan
            metrics["calibration_intercept"] = np.nan
        for name in metric_names:
            value = float(metrics[name])
            if np.isfinite(value):
                draws[name].append(value)
    intervals: dict[str, float] = {}
    for name, values in draws.items():
        if not values:
            intervals[f"{name}_ci_lower"] = np.nan
            intervals[f"{name}_ci_upper"] = np.nan
        else:
            intervals[f"{name}_ci_lower"] = float(np.quantile(values, 0.025))
            intervals[f"{name}_ci_upper"] = float(np.quantile(values, 0.975))
    intervals["bootstrap_slope_fit_failed_or_gt5_fraction"] = (
        slope_failure_or_gt5 / n_resamples
    )
    intervals["bootstrap_n_resamples"] = n_resamples
    return intervals


def _paired_brier_bootstrap_difference(
    labels: np.ndarray,
    probabilities_a0: np.ndarray,
    probabilities_a0_a1: np.ndarray,
    patient_ids: list[str],
    *,
    seed: int,
    n_resamples: int = E3_BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Bootstrap the paired target Brier difference, A0+A1 minus A0."""
    y = np.asarray(labels, dtype=int)
    p_a0 = np.asarray(probabilities_a0, dtype=float)
    p_a0_a1 = np.asarray(probabilities_a0_a1, dtype=float)
    patients = np.asarray(patient_ids, dtype=str)
    if not (len(y) == len(p_a0) == len(p_a0_a1) == len(patients)):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    if len(set(patients)) != len(y):
        raise ValueError("paired target bootstrap requires one sample per patient")
    point_difference = float(
        np.mean((y - p_a0_a1) ** 2) - np.mean((y - p_a0) ** 2)
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for draw in range(n_resamples):
        indices = rng.integers(0, len(y), len(y))
        differences[draw] = np.mean(
            (y[indices] - p_a0_a1[indices]) ** 2
            - (y[indices] - p_a0[indices]) ** 2
        )
    return {
        "target_brier_difference_a0_a1_minus_a0": point_difference,
        "target_brier_difference_a0_a1_minus_a0_ci_lower": float(
            np.quantile(differences, 0.025)
        ),
        "target_brier_difference_a0_a1_minus_a0_ci_upper": float(
            np.quantile(differences, 0.975)
        ),
    }


def final_evaluation(
    prediction_bundle: dict[str, object],
    target_label_path: str | Path,
    target_patient_ids: list[str],
    *,
    bootstrap_seed: int,
    minimum_target_positives: int = 25,
) -> dict[str, pd.DataFrame]:
    """Load target labels only after feature adaptation and predictions finish."""
    outcome = str(prediction_bundle["outcome"])
    labels_frame = pd.read_csv(
        target_label_path,
        sep="\t",
        usecols=["SAMPLE_ID", outcome],
        dtype={"SAMPLE_ID": str},
    ).set_index("SAMPLE_ID")
    target_samples = list(prediction_bundle["target_samples"])
    target_labels = labels_frame.loc[target_samples, outcome]
    profiled = target_labels.notna().to_numpy()
    target_indices = np.flatnonzero(profiled)
    y_target = target_labels.iloc[target_indices].to_numpy(dtype=int)
    if len(target_patient_ids) != len(target_samples):
        raise ValueError("target patient IDs do not align with target samples")
    if outcome == "DNMT3A" and int(y_target.sum()) < minimum_target_positives:
        print(
            "E3/E4 DNMT3A secondary outcome skipped: "
            f"{int(y_target.sum())} profiled target positives < "
            f"{minimum_target_positives}"
        )
        return {
            "cross_cohort": pd.DataFrame(),
            "repeats": pd.DataFrame(),
            "conformal": pd.DataFrame(),
            "reliability": pd.DataFrame(),
        }
    aligned_patients = [
        patient
        for patient, keep in zip(target_patient_ids, profiled)
        if keep
    ]
    return _assemble_cross_cohort_tables(
        prediction_bundle,
        y_target,
        aligned_patients,
        target_indices,
        bootstrap_seed=bootstrap_seed,
    )


def _assemble_cross_cohort_tables(
    prediction_bundle: dict[str, object],
    target_labels: np.ndarray,
    target_patient_ids: list[str],
    target_indices: np.ndarray,
    *,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    """Evaluate prepared predictions and assemble E3/E4 and conformal outputs."""
    main_entries = prediction_bundle["main"]
    repeat_entries = prediction_bundle["repeats"]
    outcome = str(prediction_bundle["outcome"])
    main_rows: list[dict[str, object]] = []
    repeat_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    repeat_target_metrics: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    repeat_source_metrics: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for entry in [*repeat_entries, *main_entries]:
        split_set = str(entry["split_set"])
        repeat = entry["repeat"]
        method = str(entry["method"])
        diagnostics = entry["diagnostics"]
        failed = bool(entry["failed"])
        for model_name in E3_MODELS:
            for calibrator_name in E3_CALIBRATORS:
                identity = {
                    "split_set": split_set,
                    "repeat": repeat,
                    "method": method,
                    "model": model_name,
                    "calibrator": calibrator_name,
                    "outcome": outcome,
                    "status": "failed" if failed else "complete",
                    "analysis_role": (
                        "confirmatory"
                        if model_name == "logistic"
                        and calibrator_name == "none"
                        and method in {"A0", "A0+A1"}
                        else "exploratory"
                    ),
                }
                if failed:
                    row = {
                        **identity,
                        **diagnostics,
                        "failure_reason": entry["failure_reason"],
                    }
                    for metric in (
                        "auroc", "auprc", "brier", "ece",
                        "calibration_intercept", "calibration_slope",
                    ):
                        row[f"target_{metric}"] = np.nan
                    if split_set == "registered_main_split":
                        main_rows.append(row)
                    else:
                        row["row_type"] = "repeat"
                        repeat_rows.append(row)
                    coverage_rows.append(
                        {
                            **identity,
                            **diagnostics,
                            "failure_reason": entry["failure_reason"],
                        }
                    )
                    continue
                model_predictions = entry["predictions"][model_name]
                calibrator = model_predictions["calibrators"][calibrator_name]
                source_cal_probabilities = _apply_calibrator(
                    calibrator_name,
                    calibrator,
                    model_predictions["source_calibration_probabilities"],
                )
                source_test_probabilities = _apply_calibrator(
                    calibrator_name,
                    calibrator,
                    model_predictions["source_test_probabilities"],
                )
                target_probabilities = _apply_calibrator(
                    calibrator_name,
                    calibrator,
                    model_predictions["target_probabilities"],
                )[target_indices]
                metrics = _prediction_metrics(target_labels, target_probabilities)
                source_test_labels = entry["source_test_labels"]
                source_metrics = _prediction_metrics(
                    source_test_labels, source_test_probabilities
                )
                if metrics["frac_pred_at_clip_bounds"] > 0.05:
                    metrics["calibration_slope"] = np.nan
                    metrics["calibration_intercept"] = np.nan
                if source_metrics["frac_pred_at_clip_bounds"] > 0.05:
                    source_metrics["calibration_slope"] = np.nan
                    source_metrics["calibration_intercept"] = np.nan
                source_calibration_labels = entry["source_calibration_labels"]
                quantile = conformal_quantile(
                    1
                    - np.where(
                        source_calibration_labels == 1,
                        source_cal_probabilities,
                        1 - source_cal_probabilities,
                    ),
                    alpha=0.1,
                )
                source_coverage = conformal_coverage_metrics(
                    source_test_labels, source_test_probabilities, quantile
                )
                target_coverage = conformal_coverage_metrics(
                    target_labels, target_probabilities, quantile
                )
                coverage_row = {
                    **identity,
                    "hyperparameters": json.dumps(
                        model_predictions["hyperparameters"], sort_keys=True
                    ),
                    "source_test_threshold": quantile,
                    **{
                        f"source_test_{key}": value
                        for key, value in source_coverage.items()
                    },
                    **{
                        f"target_{key}": value
                        for key, value in target_coverage.items()
                    },
                }
                if split_set == "registered_main_split":
                    coverage_row.update(
                        _bootstrap_coverage_intervals(
                            target_labels,
                            target_probabilities,
                            quantile,
                            target_patient_ids,
                            seed=bootstrap_seed,
                        )
                    )
                    coverage_row["bootstrap_n_resamples"] = (
                        E3_BOOTSTRAP_RESAMPLES
                    )
                coverage_rows.append(coverage_row)
                row = {
                    **identity,
                    **diagnostics,
                    "hyperparameters": json.dumps(
                        model_predictions["hyperparameters"], sort_keys=True
                    ),
                    **{f"target_{key}": value for key, value in metrics.items()},
                    "frac_pred_at_clip_bounds": metrics[
                        "frac_pred_at_clip_bounds"
                    ],
                    **{f"source_test_{key}": value for key, value in source_metrics.items()},
                    **{f"conformal_source_test_{key}": value for key, value in source_coverage.items()},
                    **{f"conformal_target_{key}": value for key, value in target_coverage.items()},
                    "source_test_conformal_threshold": quantile,
                    "failure_reason": "",
                }
                for metric in (
                    "auroc",
                    "auprc",
                    "brier",
                    "ece",
                    "calibration_intercept",
                    "calibration_slope",
                ):
                    row[f"target_minus_source_test_{metric}"] = (
                        float(metrics[metric]) - float(source_metrics[metric])
                        if np.isfinite(metrics[metric])
                        and np.isfinite(source_metrics[metric])
                        else np.nan
                    )
                if split_set == "registered_main_split":
                    row.update(
                        {
                            f"target_{key}": value
                            for key, value in _bootstrap_metric_intervals(
                                target_labels,
                                target_probabilities,
                                target_patient_ids,
                                seed=bootstrap_seed,
                            ).items()
                        }
                    )
                    row["target_bootstrap_n_resamples"] = (
                        E3_BOOTSTRAP_RESAMPLES
                    )
                    if metrics["frac_pred_at_clip_bounds"] > 0.05:
                        for metric in ("calibration_intercept", "calibration_slope"):
                            row[f"target_{metric}_ci_lower"] = np.nan
                            row[f"target_{metric}_ci_upper"] = np.nan
                if split_set == "registered_main_split":
                    main_rows.append(row)
                    if model_name == "logistic" and calibrator_name == "none":
                        reliability_rows.extend(
                            _equal_count_reliability_rows(
                                method,
                                target_labels,
                                target_probabilities,
                                outcome=outcome,
                            )
                        )
                else:
                    row["row_type"] = "repeat"
                    repeat_rows.append(row)
                    key = (method, model_name, calibrator_name)
                    repeat_target_metrics.setdefault(key, []).append(
                        {name: float(value) for name, value in metrics.items() if isinstance(value, (float, int))}
                    )
                    repeat_source_metrics.setdefault(key, []).append(
                        {
                            name: float(value)
                            for name, value in source_metrics.items()
                            if isinstance(value, (float, int))
                        }
                    )
    primary_main_entries = {
        str(entry["method"]): entry
        for entry in main_entries
        if entry["method"] in {"A0", "A0+A1"} and not entry["failed"]
    }
    if set(primary_main_entries) == {"A0", "A0+A1"}:
        primary_probabilities = {}
        for method, entry in primary_main_entries.items():
            raw_probabilities = entry["predictions"]["logistic"][
                "target_probabilities"
            ]
            primary_probabilities[method] = _apply_calibrator(
                "none", None, raw_probabilities
            )[target_indices]
        paired_difference = _paired_brier_bootstrap_difference(
            target_labels,
            primary_probabilities["A0"],
            primary_probabilities["A0+A1"],
            target_patient_ids,
            seed=bootstrap_seed,
        )
        for row in main_rows:
            if (
                row["model"] == "logistic"
                and row["calibrator"] == "none"
                and row["method"] in {"A0", "A0+A1"}
            ):
                row.update(paired_difference)
    for row in main_rows:
        key = (str(row["method"]), str(row["model"]), str(row["calibrator"]))
        metrics = repeat_target_metrics.get(key, [])
        source_metrics = repeat_source_metrics.get(key, [])
        for metric in (
            "auroc",
            "auprc",
            "brier",
            "ece",
            "calibration_intercept",
            "calibration_slope",
        ):
            values = [item[metric] for item in metrics if metric in item and np.isfinite(item[metric])]
            target_summary_values = values
            if metric in {"calibration_intercept", "calibration_slope"} and len(values) < 15:
                target_summary_values = []
            row[f"target_repeat_{metric}_n_valid_repeats"] = len(values)
            row[f"target_repeat_median_{metric}"] = (
                float(np.median(target_summary_values))
                if target_summary_values
                else np.nan
            )
            row[f"target_repeat_q1_{metric}"] = (
                float(np.quantile(target_summary_values, 0.25))
                if target_summary_values
                else np.nan
            )
            row[f"target_repeat_q3_{metric}"] = (
                float(np.quantile(target_summary_values, 0.75))
                if target_summary_values
                else np.nan
            )
            source_values = [
                item[metric]
                for item in source_metrics
                if metric in item and np.isfinite(item[metric])
            ]
            source_summary_values = source_values
            if metric in {"calibration_intercept", "calibration_slope"} and len(source_values) < 15:
                source_summary_values = []
            row[f"source_test_repeat_{metric}_n_valid_repeats"] = len(
                source_values
            )
            row[f"source_repeat_median_{metric}"] = (
                float(np.median(source_summary_values))
                if source_summary_values
                else np.nan
            )
            row[f"source_repeat_q1_{metric}"] = (
                float(np.quantile(source_summary_values, 0.25))
                if source_summary_values
                else np.nan
            )
            row[f"source_repeat_q3_{metric}"] = (
                float(np.quantile(source_summary_values, 0.75))
                if source_summary_values
                else np.nan
            )
    repeat_rows = _append_repeat_summaries(repeat_rows)
    return {
        "cross_cohort": pd.DataFrame(main_rows),
        "repeats": pd.DataFrame(repeat_rows),
        "conformal": pd.DataFrame(coverage_rows),
        "reliability": pd.DataFrame(reliability_rows),
    }


def _bootstrap_coverage_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    patient_ids: list[str],
    *,
    seed: int,
    n_resamples: int = E3_BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    patients = np.asarray(patient_ids, dtype=str)
    if len(y) != len(p) or len(y) != len(patients) or len(set(patients)) != len(y):
        raise ValueError("target coverage bootstrap requires aligned unique patients")
    rng = np.random.default_rng(seed)
    metrics = (
        "coverage",
        "positive_class_coverage",
        "negative_class_coverage",
        "mean_prediction_set_size",
        "empty_set_fraction",
    )
    values = {name: [] for name in metrics}
    for _ in range(n_resamples):
        indices = rng.integers(0, len(y), len(y))
        summary = conformal_coverage_metrics(y[indices], p[indices], threshold)
        for name in metrics:
            if np.isfinite(summary[name]):
                values[name].append(summary[name])
    return {
        bound: float(np.quantile(values[metric], q))
        for metric in metrics
        for bound, q in (
            (f"target_{metric}_ci_lower", 0.025),
            (f"target_{metric}_ci_upper", 0.975),
        )
    }


def _append_repeat_summaries(
    repeat_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Append median and IQR summaries without combining main-split results."""
    if not repeat_rows:
        return []
    frame = pd.DataFrame(repeat_rows)
    identity_columns = (
        "split_set",
        "method",
        "model",
        "calibrator",
        "outcome",
        "analysis_role",
    )
    metric_columns = [
        column
        for column in frame.columns
        if (
            column.startswith("target_")
            or column.startswith("source_test_")
            or column.startswith("target_minus_source_test_")
            or column.startswith("conformal_")
            or column
            in {
                "frac_pred_at_clip_bounds",
                "domain_classifier_auroc_cv",
                "mean_gene_ks",
                "effective_sample_size",
                "weight_min",
                "weight_median",
                "max_weight",
                "frac_weights_at_lower_clip",
                "frac_weights_at_upper_clip",
                "bootstrap_slope_fit_failed_or_gt5_fraction",
            }
        )
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summaries: list[dict[str, object]] = []
    for _, group in frame.groupby(list(identity_columns), sort=True, dropna=False):
        base = {
            column: group.iloc[0][column]
            for column in identity_columns
            if column in group.columns
        }
        for statistic, quantile in (
            ("median", 0.5),
            ("q1", 0.25),
            ("q3", 0.75),
        ):
            row: dict[str, object] = {
                **base,
                "repeat": statistic,
                "row_type": statistic,
                "n_repeats": int(group["repeat"].nunique()),
                "status": (
                    "failed"
                    if group["status"].eq("failed").all()
                    else "complete"
                ),
            }
            for column in metric_columns:
                values = (
                    pd.to_numeric(group[column], errors="coerce")
                    .astype(float)
                    .dropna()
                )
                if (
                    column.startswith(("target_", "source_test_"))
                    and column.endswith(
                        ("calibration_intercept", "calibration_slope")
                    )
                ):
                    side = (
                        "target"
                        if column.startswith("target_")
                        else "source_test"
                    )
                    metric = (
                        "calibration_intercept"
                        if column.endswith("calibration_intercept")
                        else "calibration_slope"
                    )
                    row[f"{side}_{metric}_n_valid_repeats"] = len(values)
                    if len(values) < 15:
                        values = values.iloc[0:0]
                row[column] = (
                    float(values.quantile(quantile)) if not values.empty else np.nan
                )
            summaries.append(row)
    for row in repeat_rows:
        row.setdefault("row_type", "repeat")
    return [*repeat_rows, *summaries]


def _equal_count_reliability_rows(
    method: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    outcome: str,
    bins: int = 5,
) -> list[dict[str, object]]:
    order = np.argsort(probabilities, kind="mergesort")
    return [
        {
            "method": method,
            "bin": index,
            "bin_count": int(len(indices)),
            "mean_predicted_probability": float(probabilities[indices].mean()),
            "observed_fraction": float(labels[indices].mean()),
        }
        for index, indices in enumerate(np.array_split(order, min(bins, len(order))), 1)
    ]
