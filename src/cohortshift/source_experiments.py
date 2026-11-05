"""Source-only E1 baseline and E2 calibration experiments."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .calibration import PlattScaler, fit_platt_scaler
from .features import select_top_variance_genes
from .models import (
    SOURCE_HIST_GRADIENT_BOOSTING_GRID,
    SOURCE_LOGISTIC_GRID,
    source_hist_gradient_boosting,
    source_logistic_pipeline,
)

SOURCE_EXPERIMENT_SEEDS = tuple(range(7101, 7121))
SOURCE_MAIN_SPLIT_SEED = 3101
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
        "frac_pred_at_clip_bounds": float(
            ((p <= 1e-6) | (p >= 1 - 1e-6)).mean()
        ),
    }
    if include_calibration:
        calibration_model = LogisticRegression(
            C=np.inf, solver="lbfgs", max_iter=2000
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            calibration_model.fit(_clipped_logit(p).reshape(-1, 1), y)
        metrics["calibration_slope"] = float(calibration_model.coef_[0, 0])
        metrics["calibration_intercept"] = _calibration_intercept_fixed_slope(y, p)
        metrics["calibration_slope_gt_5"] = bool(
            metrics["calibration_slope"] > 5
        )
        metrics["calibration_slope_fit_converged"] = not any(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )
        metrics["calibration_slope_fit_failed"] = not metrics[
            "calibration_slope_fit_converged"
        ]
    return metrics, bins


def _fit_calibrator(
    kind: str, calibration_probabilities: np.ndarray, calibration_labels: np.ndarray
) -> PlattScaler | IsotonicRegression | None:
    if kind == "none":
        return None
    if kind == "Platt":
        return fit_platt_scaler(calibration_probabilities, calibration_labels)
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(calibration_probabilities, dtype=float),
            np.asarray(calibration_labels, dtype=int),
        )
    raise ValueError(f"Unknown calibration method: {kind}")


def _apply_calibrator(
    kind: str,
    calibrator: PlattScaler | IsotonicRegression | None,
    probabilities: np.ndarray,
) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if kind == "none":
        return np.clip(p, 1e-6, 1 - 1e-6)
    if isinstance(calibrator, PlattScaler):
        return np.clip(
            calibrator.predict_proba(_clipped_logit(p))[:, 1],
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
    train_expression: pd.DataFrame,
    train_labels: np.ndarray,
    *,
    seed: int,
    scale_logistic: bool = True,
    sample_weights: np.ndarray | None = None,
) -> dict[str, tuple[object, dict[str, object]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 2000)
    logistic_estimator = (
        source_logistic_pipeline(random_state=seed)
        if scale_logistic
        else LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
    )
    logistic_grid = (
        SOURCE_LOGISTIC_GRID
        if scale_logistic
        else {"C": SOURCE_LOGISTIC_GRID["classifier__C"]}
    )
    searches = {
        "logistic": (
            logistic_estimator,
            logistic_grid,
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
        fit_parameters = {}
        if sample_weights is not None:
            fit_parameters[
                "classifier__sample_weight"
                if name == "logistic" and scale_logistic
                else "sample_weight"
            ] = sample_weights
        search.fit(train_expression, train_labels, **fit_parameters)
        fitted[name] = (search.best_estimator_, search.best_params_)
    return fitted


def _summary_rows(
    rows: list[dict[str, object]],
    *,
    group_columns: tuple[str, ...],
    metric_names: tuple[str, ...],
    statistics: tuple[str, ...] = ("mean", "sd"),
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    summaries = []
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for summary_stat in statistics:
            row: dict[str, object] = {
                **base,
                "row_type": summary_stat,
                "repeat": summary_stat,
                "split_set": "20_repeats",
                "included_in_20_repeats": True,
                "seed": "",
                "hyperparameters": "",
            }
            for metric in metric_names:
                values = group[metric].astype(float)
                values = values.loc[np.isfinite(values)]
                if values.empty:
                    row[metric] = np.nan
                    continue
                if metric in {"calibration_slope", "calibration_intercept"} and summary_stat in {
                    "mean",
                    "sd",
                }:
                    row[metric] = np.nan
                elif summary_stat == "mean":
                    row[metric] = float(values.mean())
                elif summary_stat == "sd":
                    row[metric] = float(values.std(ddof=1))
                elif summary_stat == "median":
                    row[metric] = float(values.median())
                elif summary_stat == "q1":
                    row[metric] = float(values.quantile(0.25))
                else:
                    row[metric] = float(values.quantile(0.75))
            row["ece_bins"] = 5
            row["ece_bin_counts"] = ""
            row["slope_gt_5_repeats"] = int(
                (group["calibration_slope"] > 5).sum()
            ) if "calibration_slope" in group else np.nan
            row["slope_fit_not_converged_repeats"] = int(
                (~group["calibration_slope_fit_converged"].astype(bool)).sum()
            ) if "calibration_slope_fit_converged" in group else np.nan
            summaries.append(row)
    return summaries


def _paired_difference_rows(
    e2_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    frame = pd.DataFrame(e2_rows)
    repeated = frame.loc[frame["split_set"] == "20_repeats"]
    none = repeated.loc[repeated["calibrator"] == "none"].set_index(
        ["model", "repeat"]
    )
    paired_rows = []
    for row in repeated.loc[repeated["calibrator"] != "none"].to_dict("records"):
        baseline = none.loc[(row["model"], row["repeat"])]
        paired_rows.append(
            {
                **{
                    key: row[key]
                    for key in ("repeat", "seed", "model", "calibrator")
                },
                "row_type": "paired_difference",
                "split_set": "20_repeats",
                "included_in_20_repeats": True,
                "brier_difference": float(row["brier"] - baseline["brier"]),
                "ece_difference": float(row["ece"] - baseline["ece"]),
            }
        )
    paired = pd.DataFrame(paired_rows)
    summaries = []
    for (model, calibrator), group in paired.groupby(
        ["model", "calibrator"], sort=True
    ):
        for statistic in ("mean", "sd"):
            summaries.append(
                {
                    "row_type": f"paired_difference_{statistic}",
                    "split_set": "20_repeats",
                    "included_in_20_repeats": True,
                    "repeat": statistic,
                    "seed": "",
                    "model": model,
                    "calibrator": calibrator,
                    "brier_difference": float(
                        group["brier_difference"].mean()
                        if statistic == "mean"
                        else group["brier_difference"].std(ddof=1)
                    ),
                    "ece_difference": float(
                        group["ece_difference"].mean()
                        if statistic == "mean"
                        else group["ece_difference"].std(ddof=1)
                    ),
                }
            )
    return [*paired_rows, *summaries]


def run_source_experiments(
    source_expression: pd.DataFrame,
    source_labels: pd.Series,
    patient_by_sample: Mapping[str, str],
    *,
    seeds: Sequence[int] = SOURCE_EXPERIMENT_SEEDS,
    n_genes: int = 2000,
    registered_main_split: Mapping[str, str] | None = None,
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

    split_specs: list[
        tuple[int | str, int, dict[str, list[str]], str, bool]
    ] = []
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
        split_specs.append(
            (repeat, int(seed), sample_splits, "20_repeats", True)
        )

    if registered_main_split is not None:
        if set(registered_main_split) != set(source_expression.index):
            raise ValueError(
                "Registered main split sample IDs must match source expression"
            )
        if not set(registered_main_split.values()).issubset(
            {"train", "validation", "test"}
        ):
            raise ValueError(
                "Registered main split values must be train, validation, or test"
            )
        registered_samples = {
            "train": sorted(
                sample
                for sample, split in registered_main_split.items()
                if split == "train"
            ),
            "calibration": sorted(
                sample
                for sample, split in registered_main_split.items()
                if split == "validation"
            ),
            "test": sorted(
                sample
                for sample, split in registered_main_split.items()
                if split == "test"
            ),
        }
        split_specs.append(
            (
                "registered_main_split",
                SOURCE_MAIN_SPLIT_SEED,
                registered_samples,
                "registered_main_split",
                False,
            )
        )

    for repeat, seed, sample_splits, split_set, included_in_20_repeats in split_specs:
        for split, sample_ids in sample_splits.items():
            split_rows.extend(
                {
                    "repeat": repeat,
                    "seed": int(seed),
                    "sample_id": sample,
                    "patient_id": patient_by_sample[sample],
                    "split": (
                        "validation"
                        if split == "calibration"
                        and split_set == "registered_main_split"
                        else split
                    ),
                    "split_set": split_set,
                    "included_in_20_repeats": included_in_20_repeats,
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
                "split_set": split_set,
                "included_in_20_repeats": included_in_20_repeats,
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
                    "row_type": (
                        "repeat" if split_set == "20_repeats" else split_set
                    ),
                    "repeat": repeat,
                    "seed": int(seed),
                    "split_set": split_set,
                    "included_in_20_repeats": included_in_20_repeats,
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
                if metrics["frac_pred_at_clip_bounds"] > 0.05:
                    metrics["calibration_slope"] = np.nan
                    metrics["calibration_intercept"] = np.nan
                e2_rows.append(
                    {
                        "row_type": (
                            "repeat" if split_set == "20_repeats" else split_set
                        ),
                        "repeat": repeat,
                        "seed": int(seed),
                        "split_set": split_set,
                        "included_in_20_repeats": included_in_20_repeats,
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
                        "split_set": split_set,
                        "included_in_20_repeats": included_in_20_repeats,
                        "model": model_name,
                        "calibrator": calibrator_name,
                        **bin_row,
                    }
                    for bin_row in bins
                )

    e1_metric_names = ("auroc", "auprc", "brier", "ece")
    e1_summary = _summary_rows(
        [row for row in e1_rows if row["split_set"] == "20_repeats"],
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
        "frac_pred_at_clip_bounds",
    )
    e2_summary = _summary_rows(
        [row for row in e2_rows if row["split_set"] == "20_repeats"],
        group_columns=("model", "calibrator"),
        metric_names=e2_metric_names,
        statistics=("mean", "sd", "median", "q1", "q3"),
    )
    for row in e2_summary:
        repeated_group = pd.DataFrame(
            [
                result
                for result in e2_rows
                if result["split_set"] == "20_repeats"
                and result["model"] == row["model"]
                and result["calibrator"] == row["calibrator"]
            ]
        )
        row["slope_gt_5_repeats"] = int(
            repeated_group["calibration_slope_gt_5"].sum()
        )
        row["slope_fit_not_converged_repeats"] = int(
            (~repeated_group["calibration_slope_fit_converged"]).sum()
        )
        row["slope_gt_5_or_fit_not_converged_repeats"] = int(
            (
                repeated_group["calibration_slope_gt_5"]
                | ~repeated_group["calibration_slope_fit_converged"]
            ).sum()
        )
    paired_rows = _paired_difference_rows(e2_rows)
    return {
        "e1": pd.DataFrame([*e1_rows, *e1_summary]),
        "e2": pd.DataFrame([*e2_rows, *e2_summary, *paired_rows]),
        "reliability": pd.DataFrame(reliability_rows),
        "splits": pd.DataFrame(split_rows),
        "selected_genes": pd.DataFrame(selected_gene_rows),
    }


def save_source_reliability_plot(
    reliability_data: pd.DataFrame, output_path: str
) -> None:
    """Plot source-only reliability curves aggregated across repeats."""
    reliability_data = reliability_data.loc[
        reliability_data["split_set"] == "20_repeats"
    ]
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
