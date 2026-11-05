"""Small orchestration helpers for pre-registered method evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .adapt import adaptation_weights, adapt_features
from .conformal import conformal_coverage_metrics, conformal_quantile
from .features import select_top_variance_genes
from .models import (
    SOURCE_HIST_GRADIENT_BOOSTING_GRID,
    SOURCE_LOGISTIC_GRID,
    source_hist_gradient_boosting,
    source_logistic_pipeline,
)
from .shift import domain_classifier_auroc, per_gene_ks
from .source_experiments import (
    _apply_calibrator,
    _fit_calibrator,
    _metric_values,
)

E3_METHODS = ("raw", "A0", "A0+A1", "A2", "A0+A3")
E3_CALIBRATORS = ("none", "Platt", "isotonic")
E3_MODELS = ("logistic", "hist_gradient_boosting")
E3_BOOTSTRAP_RESAMPLES = 2000


def _tune_e3_models(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    *,
    seed: int,
    avoid_second_scaler: bool,
) -> dict[str, tuple[object, dict[str, object]]]:
    folds = min(5, int(np.bincount(np.asarray(train_labels, dtype=int)).min()))
    if folds < 2:
        raise ValueError("source training split needs at least two samples per class")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    logistic = (
        LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=seed
        )
        if avoid_second_scaler
        else source_logistic_pipeline(random_state=seed)
    )
    logistic_grid = (
        {"C": SOURCE_LOGISTIC_GRID["classifier__C"]}
        if avoid_second_scaler
        else SOURCE_LOGISTIC_GRID
    )
    estimators: dict[str, tuple[object, dict[str, object]]] = {}
    searches = {
        "logistic": (logistic, logistic_grid),
        "hist_gradient_boosting": (
            source_hist_gradient_boosting(random_state=seed),
            SOURCE_HIST_GRADIENT_BOOSTING_GRID,
        ),
    }
    for name, (estimator, grid) in searches.items():
        search = GridSearchCV(
            estimator,
            grid,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(train_features, train_labels)
        estimators[name] = (search.best_estimator_, search.best_params_)
    return estimators


def _fit_e3_estimator(
    estimator: object,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    weights: np.ndarray | None,
) -> object:
    fitted = clone(estimator)
    if weights is None:
        fitted.fit(train_features, train_labels)
    elif isinstance(fitted, Pipeline):
        fitted.fit(
            train_features,
            train_labels,
            classifier__sample_weight=weights,
        )
    else:
        fitted.fit(train_features, train_labels, sample_weight=weights)
    return fitted


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
    if method == "A0+A3":
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
            "failure_reason": (
                "A3 degenerate; registered classifier and clip were not changed"
                if reasons
                else ""
            ),
        }
        if reasons:
            return {
                "failed": True,
                "failure_reason": weight_diagnostics["failure_reason"],
                "weight_diagnostics": weight_diagnostics,
            }
        weights = weights / weights.mean()
    estimators = _tune_e3_models(
        train_x,
        source_labels,
        seed=seed,
        avoid_second_scaler=(method in {"A1", "A0+A1"}),
    )
    fitted_models: dict[str, object] = {}
    predictions: dict[str, object] = {}
    for name, (estimator, parameters) in estimators.items():
        fitted = _fit_e3_estimator(estimator, train_x, source_labels, weights)
        fitted_models[name] = fitted
        calibration_probability = fitted.predict_proba(calibration_x)[:, 1]
        test_probability = fitted.predict_proba(test_x)[:, 1]
        target_probability = fitted.predict_proba(target_x)[:, 1]
        calibrators = {
            kind: _fit_calibrator(kind, calibration_probability, calibration_labels)
            for kind in E3_CALIBRATORS
        }
        predictions[name] = {
            "hyperparameters": parameters,
            "estimator": fitted,
            "calibrators": calibrators,
            "source_calibration_probabilities": calibration_probability,
            "source_test_probabilities": test_probability,
            "target_probabilities": target_probability,
        }
    return {
        "failed": False,
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
    source_main = source_expression.loc[:, registered_main_genes]
    target_main = target_expression.loc[:, registered_main_genes]
    domain_diagnostics: dict[str, dict[str, float | str]] = {}
    for method in E3_METHODS:
        domain_method = "none" if method == "raw" else method
        domain_auc = domain_classifier_auroc(
            source_main, target_main, method=domain_method, seed=seed
        )
        gene_ks = per_gene_ks(source_main, target_main, method=domain_method)
        domain_diagnostics[method] = {
            "domain_classifier_auroc_cv": float(domain_auc),
            "mean_gene_ks": float(gene_ks["ks_statistic"].mean()),
            "domain_classifier_auroc_note": (
                "unweighted; unchanged by construction (A3 weights are used for "
                "model training)"
                if method == "A0+A3"
                else "unweighted"
            ),
        }

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
        train_y = source_labels.loc[train_ids].to_numpy(dtype=int)
        calibration_y = source_labels.loc[calibration_ids].to_numpy(dtype=int)
        test_y = source_labels.loc[test_ids].to_numpy(dtype=int)
        for method in E3_METHODS:
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
    if repeats.empty:
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
        "domain_diagnostics": domain_diagnostics,
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
    metric_names = ("auroc", "auprc", "brier", "ece", "calibration_intercept", "calibration_slope")
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(n_resamples):
        sampled_patients = rng.integers(0, len(unique_patients), len(unique_patients))
        indices = np.concatenate(
            [np.flatnonzero(patient_index == index) for index in sampled_patients]
        )
        if not len(indices) or len(np.unique(y[indices])) != 2:
            continue
        metrics = _prediction_metrics(y[indices], p[indices])
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
    return intervals


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
                        coverage_rows.append(
                            {
                                **identity,
                                **diagnostics,
                                "failure_reason": entry["failure_reason"],
                            }
                        )
                    else:
                        repeat_rows.append(row)
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
                if (
                    calibrator_name in {"Platt", "isotonic"}
                    and metrics["frac_pred_at_clip_bounds"] > 0.05
                ):
                    metrics["calibration_slope"] = np.nan
                    metrics["calibration_intercept"] = np.nan
                if (
                    calibrator_name in {"Platt", "isotonic"}
                    and source_metrics["frac_pred_at_clip_bounds"] > 0.05
                ):
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
                if (
                    split_set == "registered_main_split"
                    and model_name == "logistic"
                    and calibrator_name == "none"
                ):
                    coverage_row.update(
                        _bootstrap_coverage_intervals(
                            target_labels,
                            target_probabilities,
                            quantile,
                            target_patient_ids,
                            seed=bootstrap_seed,
                        )
                    )
                coverage_rows.append(coverage_row)
                row = {
                    **identity,
                    **diagnostics,
                    "hyperparameters": json.dumps(
                        model_predictions["hyperparameters"], sort_keys=True
                    ),
                    **{f"target_{key}": value for key, value in metrics.items()},
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
                if (
                    split_set == "registered_main_split"
                    and model_name == "logistic"
                    and calibrator_name == "none"
                ):
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
    for row in main_rows:
        key = (str(row["method"]), str(row["model"]), str(row["calibrator"]))
        metrics = repeat_target_metrics.get(key, [])
        source_metrics = repeat_source_metrics.get(key, [])
        for metric in ("auroc", "auprc", "brier", "ece", "calibration_intercept", "calibration_slope"):
            values = [item[metric] for item in metrics if metric in item and np.isfinite(item[metric])]
            row[f"target_repeat_mean_{metric}"] = float(np.mean(values)) if values else np.nan
            row[f"target_repeat_sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"target_repeat_median_{metric}"] = float(np.median(values)) if values else np.nan
            row[f"target_repeat_q1_{metric}"] = float(np.quantile(values, 0.25)) if values else np.nan
            row[f"target_repeat_q3_{metric}"] = float(np.quantile(values, 0.75)) if values else np.nan
            source_values = [
                item[metric]
                for item in source_metrics
                if metric in item and np.isfinite(item[metric])
            ]
            row[f"source_repeat_mean_{metric}"] = (
                float(np.mean(source_values)) if source_values else np.nan
            )
            row[f"source_repeat_sd_{metric}"] = (
                float(np.std(source_values, ddof=1))
                if len(source_values) > 1
                else np.nan
            )
            row[f"source_repeat_median_{metric}"] = (
                float(np.median(source_values)) if source_values else np.nan
            )
            row[f"source_repeat_q1_{metric}"] = (
                float(np.quantile(source_values, 0.25)) if source_values else np.nan
            )
            row[f"source_repeat_q3_{metric}"] = (
                float(np.quantile(source_values, 0.75)) if source_values else np.nan
            )
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
