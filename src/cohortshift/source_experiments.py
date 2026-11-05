"""Source-only E1 baseline and E2 calibration experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .features import select_top_variance_genes
from .models import (
    SOURCE_HIST_GRADIENT_BOOSTING_GRID,
    SOURCE_LOGISTIC_GRID,
    source_hist_gradient_boosting,
    source_logistic_pipeline,
)

SOURCE_EXPERIMENT_SEEDS = tuple(range(7101, 7121))
CALIBRATORS = ("none", "Platt", "isotonic")


def select_repeat_genes(
    source_expression: pd.DataFrame,
    training_sample_ids: Sequence[str],
    *,
    n_genes: int = 2000,
) -> list[str]:
    """Select variance-ranked genes from source training rows only."""
    missing_samples = set(training_sample_ids) - set(source_expression.index)
    if missing_samples:
        raise ValueError(
            "Source training samples are missing from the expression matrix: "
            + ", ".join(sorted(missing_samples)[:5])
        )
    training_expression = source_expression.loc[list(training_sample_ids)]
    return select_top_variance_genes(training_expression, n_genes=n_genes)


def _probability_bins(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> tuple[float, list[dict[str, float | int]]]:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("labels and probabilities must be non-empty matching vectors")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    order = np.argsort(p, kind="mergesort")
    bin_rows = []
    for bin_index, indices in enumerate(np.array_split(order, min(n_bins, len(order))), 1):
        bin_rows.append(
            {
                "bin": bin_index,
                "bin_count": int(len(indices)),
                "mean_predicted_probability": float(p[indices].mean()),
                "observed_fraction": float(y[indices].mean()),
            }
        )
    ece = sum(
        row["bin_count"]
        / len(y)
        * abs(
            row["observed_fraction"] - row["mean_predicted_probability"]
        )
        for row in bin_rows
    )
    return float(ece), bin_rows


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 5
) -> tuple[float, list[dict[str, float | int]]]:
    """Return equal-count ECE and the reliability data for each bin."""
    return _probability_bins(labels, probabilities, n_bins)


def _clipped_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _calibration_intercept_fixed_slope(
    labels: np.ndarray, probabilities: np.ndarray
) -> float:
    y = np.asarray(labels, dtype=int)
    logits = _clipped_logit(probabilities)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("calibration metrics require both label classes")

    def score(intercept: float) -> float:
        return float(np.sum(expit(logits + intercept)) - y.sum())

    return float(brentq(score, -50.0, 50.0))


def _metric_values(
    labels: np.ndarray, probabilities: np.ndarray, *, include_calibration: bool
) -> tuple[dict[str, float | int | str], list[dict[str, float | int]]]:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("evaluation metrics require both label classes")
    ece, bins = expected_calibration_error(y, p, n_bins=5)
    metrics: dict[str, float | int | str] = {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece,
        "ece_bins": len(bins),
        "ece_bin_counts": "|".join(str(row["bin_count"]) for row in bins),
    }
    if include_calibration:
        calibration_model = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=2000
        ).fit(_clipped_logit(p).reshape(-1, 1), y)
        metrics["calibration_slope"] = float(calibration_model.coef_[0, 0])
        metrics["calibration_intercept"] = _calibration_intercept_fixed_slope(y, p)
    return metrics, bins


def _fit_calibrator(
    kind: str, calibration_probabilities: np.ndarray, calibration_labels: np.ndarray
) -> LogisticRegression | IsotonicRegression | None:
    if kind == "none":
        return None
    if kind == "Platt":
        return LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(
            _clipped_logit(calibration_probabilities).reshape(-1, 1),
            np.asarray(calibration_labels, dtype=int),
        )
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(calibration_probabilities, dtype=float),
            np.asarray(calibration_labels, dtype=int),
        )
    raise ValueError(f"Unknown calibration method: {kind}")


def _apply_calibrator(
    kind: str,
    calibrator: LogisticRegression | IsotonicRegression | None,
    probabilities: np.ndarray,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if kind == "none":
        return np.clip(p, 1e-6, 1 - 1e-6)
    if isinstance(calibrator, LogisticRegression):
        return np.clip(
            calibrator.predict_proba(_clipped_logit(p).reshape(-1, 1))[:, 1],
            1e-6,
            1 - 1e-6,
        )
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(p), 1e-6, 1 - 1e-6)
    raise RuntimeError(f"Calibrator {kind!r} was not fitted")


def _patient_splits(
    labels_by_patient: Mapping[str, int], *, seed: int
) -> dict[str, set[str]]:
    patient_ids = np.array(sorted(labels_by_patient))
    labels = np.array([labels_by_patient[patient] for patient in patient_ids])
    train_ids, holdout_ids, train_labels, holdout_labels = train_test_split(
        patient_ids,
        labels,
        test_size=0.4,
        random_state=seed,
        stratify=labels,
    )
    calibration_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.5,
        random_state=seed + 1000,
        stratify=holdout_labels,
    )
    return {
        "train": set(train_ids.tolist()),
        "calibration": set(calibration_ids.tolist()),
        "test": set(test_ids.tolist()),
    }


def _tuned_estimators(
    train_expression: pd.DataFrame, train_labels: np.ndarray, *, seed: int
) -> dict[str, tuple[object, dict[str, object]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 2000)
    searches = {
        "logistic": (
            source_logistic_pipeline(random_state=seed),
            SOURCE_LOGISTIC_GRID,
        ),
        "hist_gradient_boosting": (
            source_hist_gradient_boosting(random_state=seed),
            SOURCE_HIST_GRADIENT_BOOSTING_GRID,
        ),
    }
    fitted = {}
    for name, (estimator, parameter_grid) in searches.items():
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(train_expression, train_labels)
        fitted[name] = (search.best_estimator_, search.best_params_)
    return fitted


def _summary_rows(
    rows: list[dict[str, object]],
    *,
    group_columns: tuple[str, ...],
    metric_names: tuple[str, ...],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries = []
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for summary_stat in ("mean", "sd"):
            row: dict[str, object] = {
                **base,
                "row_type": summary_stat,
                "repeat": summary_stat,
                "seed": "",
                "hyperparameters": "",
            }
            for metric in metric_names:
                values = group[metric].astype(float)
                row[metric] = (
                    float(values.mean())
                    if summary_stat == "mean"
                    else float(values.std(ddof=1))
                )
            row["ece_bins"] = 5
            row["ece_bin_counts"] = ""
            summaries.append(row)
    return summaries


def run_source_experiments(
    source_expression: pd.DataFrame,
    source_labels: pd.Series,
    patient_by_sample: Mapping[str, str],
    *,
    seeds: Sequence[int] = SOURCE_EXPERIMENT_SEEDS,
    n_genes: int = 2000,
) -> dict[str, pd.DataFrame]:
    """Run repeated source-only E1/E2 splits; accepts no target-cohort inputs."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("experiment seeds must be non-empty and unique")
    if source_expression.index.has_duplicates or source_labels.index.has_duplicates:
        raise ValueError("source sample IDs must be unique")
    if set(source_expression.index) != set(source_labels.index):
        raise ValueError("source expression and NPM1 label sample IDs must match")
    if set(source_expression.index) != set(patient_by_sample):
        raise ValueError("source expression and patient-list sample IDs must match")
    if source_labels.isna().any() or not set(source_labels.unique()).issubset({0, 1}):
        raise ValueError("source NPM1 labels must be fully profiled binary values")
    if len(set(patient_by_sample.values())) != len(source_expression):
        raise ValueError("the source main set must contain one sample per patient")

    labels_by_patient = {
        patient_by_sample[sample]: int(source_labels.loc[sample])
        for sample in source_expression.index
    }
    e1_rows: list[dict[str, object]] = []
    e2_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    selected_gene_rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(seeds, start=1):
        patient_splits = _patient_splits(labels_by_patient, seed=int(seed))
        sample_splits = {
            split: sorted(
                sample
                for sample, patient in patient_by_sample.items()
                if patient in patients
            )
            for split, patients in patient_splits.items()
        }
        for split, sample_ids in sample_splits.items():
            split_rows.extend(
                {
                    "repeat": repeat,
                    "seed": int(seed),
                    "sample_id": sample,
                    "patient_id": patient_by_sample[sample],
                    "split": split,
                }
                for sample in sample_ids
            )

        training_samples = sample_splits["train"]
        selected_genes = select_repeat_genes(
            source_expression, training_samples, n_genes=n_genes
        )
        if len(selected_genes) != n_genes:
            raise ValueError(
                f"Expected {n_genes} source training genes, found {len(selected_genes)}"
            )
        selected_gene_rows.extend(
            {
                "repeat": repeat,
                "seed": int(seed),
                "gene": gene,
            }
            for gene in selected_genes
        )
        train_x = source_expression.loc[training_samples, selected_genes]
        train_y = source_labels.loc[training_samples].to_numpy(dtype=int)
        calibration_samples = sample_splits["calibration"]
        calibration_x = source_expression.loc[calibration_samples, selected_genes]
        calibration_y = source_labels.loc[calibration_samples].to_numpy(dtype=int)
        test_samples = sample_splits["test"]
        test_x = source_expression.loc[test_samples, selected_genes]
        test_y = source_labels.loc[test_samples].to_numpy(dtype=int)

        estimators = _tuned_estimators(train_x, train_y, seed=int(seed))
        for model_name, (estimator, parameters) in estimators.items():
            test_probabilities = estimator.predict_proba(test_x)[:, 1]
            e1_metrics, _ = _metric_values(
                test_y, test_probabilities, include_calibration=False
            )
            e1_rows.append(
                {
                    "row_type": "repeat",
                    "repeat": repeat,
                    "seed": int(seed),
                    "model": model_name,
                    "hyperparameters": json.dumps(parameters, sort_keys=True),
                    "n_train": len(training_samples),
                    "n_calibration": len(calibration_samples),
                    "n_test": len(test_samples),
                    **e1_metrics,
                }
            )

            calibration_probabilities = estimator.predict_proba(calibration_x)[:, 1]
            for calibrator_name in CALIBRATORS:
                calibrator = _fit_calibrator(
                    calibrator_name, calibration_probabilities, calibration_y
                )
                calibrated_test = _apply_calibrator(
                    calibrator_name, calibrator, test_probabilities
                )
                metrics, bins = _metric_values(
                    test_y, calibrated_test, include_calibration=True
                )
                e2_rows.append(
                    {
                        "row_type": "repeat",
                        "repeat": repeat,
                        "seed": int(seed),
                        "model": model_name,
                        "calibrator": calibrator_name,
                        "hyperparameters": json.dumps(parameters, sort_keys=True),
                        "n_train": len(training_samples),
                        "n_calibration": len(calibration_samples),
                        "n_test": len(test_samples),
                        **metrics,
                    }
                )
                reliability_rows.extend(
                    {
                        "repeat": repeat,
                        "seed": int(seed),
                        "model": model_name,
                        "calibrator": calibrator_name,
                        **bin_row,
                    }
                    for bin_row in bins
                )

    e1_metric_names = ("auroc", "auprc", "brier", "ece")
    e1_summary = _summary_rows(
        e1_rows,
        group_columns=("model",),
        metric_names=e1_metric_names,
    )
    e2_metric_names = (
        "auroc",
        "auprc",
        "brier",
        "ece",
        "calibration_intercept",
        "calibration_slope",
    )
    e2_summary = _summary_rows(
        e2_rows,
        group_columns=("model", "calibrator"),
        metric_names=e2_metric_names,
    )
    return {
        "e1": pd.DataFrame([*e1_rows, *e1_summary]),
        "e2": pd.DataFrame([*e2_rows, *e2_summary]),
        "reliability": pd.DataFrame(reliability_rows),
        "splits": pd.DataFrame(split_rows),
        "selected_genes": pd.DataFrame(selected_gene_rows),
    }


def save_source_reliability_plot(
    reliability_data: pd.DataFrame, output_path: str
) -> None:
    """Plot source-only reliability curves aggregated across repeats."""
    model_names = sorted(reliability_data["model"].unique())
    figure, axes = plt.subplots(
        1, len(model_names), figsize=(6 * len(model_names), 5), squeeze=False
    )
    colors = {"none": "black", "Platt": "tab:blue", "isotonic": "tab:orange"}
    for axis, model_name in zip(axes[0], model_names):
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Ideal")
        for calibrator_name in CALIBRATORS:
            curves = reliability_data.loc[
                (reliability_data["model"] == model_name)
                & (reliability_data["calibrator"] == calibrator_name)
            ]
            grouped = curves.groupby("bin", sort=True).agg(
                mean_predicted=("mean_predicted_probability", "mean"),
                observed=("observed_fraction", "mean"),
                observed_sd=("observed_fraction", "std"),
            )
            axis.errorbar(
                grouped["mean_predicted"],
                grouped["observed"],
                yerr=grouped["observed_sd"].fillna(0),
                marker="o",
                capsize=2,
                color=colors[calibrator_name],
                label=calibrator_name,
            )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted probability",
            ylabel="Observed fraction",
            title=model_name.replace("_", " "),
        )
        axis.legend()
    figure.tight_layout()
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
