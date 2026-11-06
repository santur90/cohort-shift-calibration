"""E5: few-label target recalibration of source-trained probabilities.

Recalibrators see only the k drawn target samples; those samples are excluded
from evaluation, and each draw is used exactly once.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from .calibration import fit_platt_scaler
from .experiments import _prediction_metrics
from .source_experiments import _apply_calibrator, _clipped_logit

E5_LABEL_BUDGETS = (0, 10, 20, 40)
E5_N_DRAWS = 200
E5_METHODS = ("intercept_only", "Platt")
E5_METRICS = ("brier", "calibration_intercept", "calibration_slope")


def draw_stratified_indices(
    labels: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw k samples with proportional label allocation, >= 1 per class."""
    y = np.asarray(labels, dtype=int)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if k < 2 or k > len(y) - 2:
        raise ValueError("k must leave at least two samples for evaluation")
    n_positive = int(round(k * len(positives) / len(y)))
    n_positive = min(max(n_positive, 1), k - 1, len(positives))
    n_negative = k - n_positive
    if n_negative > len(negatives):
        raise ValueError("not enough negatives for a stratified draw")
    return np.concatenate(
        (
            rng.choice(positives, n_positive, replace=False),
            rng.choice(negatives, n_negative, replace=False),
        )
    )


def _fit_intercept_only(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Offset b with slope fixed at 1 such that sum(expit(logit(p) + b)) = sum(y)."""
    y = np.asarray(labels, dtype=int)
    logits = _clipped_logit(probabilities)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("recalibration requires both label classes")
    return float(
        brentq(lambda b: float(expit(logits + b).sum() - y.sum()), -50.0, 50.0)
    )


def _recalibrate(
    method: str, drawn_p: np.ndarray, drawn_y: np.ndarray, new_p: np.ndarray
) -> np.ndarray:
    if method == "intercept_only":
        shift = _fit_intercept_only(drawn_p, drawn_y)
        return np.clip(expit(_clipped_logit(new_p) + shift), 1e-6, 1 - 1e-6)
    if method == "Platt":
        scaler = fit_platt_scaler(drawn_p, drawn_y)
        return _apply_calibrator("Platt", scaler, new_p)
    raise ValueError(f"unknown recalibration method: {method}")


def _evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    metrics = _prediction_metrics(labels, probabilities)
    if metrics["frac_pred_at_clip_bounds"] > 0.05:
        metrics["calibration_slope"] = np.nan
        metrics["calibration_intercept"] = np.nan
    return metrics


def run_few_label_recalibration(
    sample_ids: Sequence[str],
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    setting: str,
    seed: int,
    budgets: Sequence[int] = E5_LABEL_BUDGETS,
    n_draws: int = E5_N_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-draw rows, summary rows) for one adaptation setting.

    `probabilities` are the frozen source-trained target probabilities; labels
    are used only to draw, fit the recalibrator, and score held-out samples.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=int)
    ids = np.asarray(sample_ids, dtype=str)
    if not (len(p) == len(y) == len(ids)):
        raise ValueError("sample IDs, probabilities, and labels must align")
    draw_rows: list[dict[str, object]] = []
    for k in budgets:
        if k == 0:
            metrics = _evaluate(y, p)
            draw_rows.append(
                {
                    "setting": setting, "k": 0, "draw": 0, "method": "none",
                    "drawn_samples": "", "n_eval": len(y), "fit_failed": False,
                    "slope_fit_failed": bool(metrics["calibration_slope_fit_failed"]),
                    "frac_pred_at_clip_bounds": metrics["frac_pred_at_clip_bounds"],
                    **{name: float(metrics[name]) for name in E5_METRICS},
                }
            )
            continue
        rng = np.random.default_rng([seed, k])
        for draw in range(n_draws):
            drawn = draw_stratified_indices(y, k, rng)
            held_out = np.setdiff1d(np.arange(len(y)), drawn)
            baseline = _evaluate(y[held_out], p[held_out])
            baseline_row: dict[str, object] = {
                "setting": setting, "k": k, "draw": draw, "method": "none",
                "drawn_samples": ";".join(ids[drawn]), "n_eval": len(held_out),
                "fit_failed": False,
                "slope_fit_failed": bool(baseline["calibration_slope_fit_failed"]),
                "frac_pred_at_clip_bounds": baseline["frac_pred_at_clip_bounds"],
                **{name: float(baseline[name]) for name in E5_METRICS},
                **{
                    f"{name}_difference_vs_none": (
                        0.0
                        if np.isfinite(baseline[name])
                        and not (
                            name == "calibration_slope"
                            and baseline["calibration_slope_fit_failed"]
                        )
                        else np.nan
                    )
                    for name in E5_METRICS
                },
            }
            draw_rows.append(baseline_row)
            for method in E5_METHODS:
                row: dict[str, object] = {
                    "setting": setting, "k": k, "draw": draw, "method": method,
                    "drawn_samples": ";".join(ids[drawn]), "n_eval": len(held_out),
                    "fit_failed": False, "slope_fit_failed": False,
                    "frac_pred_at_clip_bounds": np.nan,
                    **{name: np.nan for name in E5_METRICS},
                }
                try:
                    recalibrated = _recalibrate(method, p[drawn], y[drawn], p[held_out])
                    metrics = _evaluate(y[held_out], recalibrated)
                    row.update(
                        {name: float(metrics[name]) for name in E5_METRICS}
                    )
                    row["frac_pred_at_clip_bounds"] = metrics["frac_pred_at_clip_bounds"]
                    row["slope_fit_failed"] = bool(metrics["calibration_slope_fit_failed"])
                    row["fit_failed"] = row["slope_fit_failed"]
                    if not row["fit_failed"]:
                        for name in E5_METRICS:
                            if (
                                name == "calibration_slope"
                                and baseline["calibration_slope_fit_failed"]
                            ):
                                continue
                            row[f"{name}_difference_vs_none"] = (
                                float(metrics[name]) - float(baseline[name])
                            )
                except (ValueError, RuntimeError, FloatingPointError):
                    row["fit_failed"] = True
                draw_rows.append(row)
    draws = pd.DataFrame(draw_rows)
    summary_rows = []
    for (k, method), group in draws.groupby(["k", "method"], sort=True):
        row = {
            "setting": setting, "k": k, "method": method,
            "n_draws": len(group),
            "median_n_eval": float(group["n_eval"].median()),
            "fraction_fit_failed": float(group["fit_failed"].mean()),
            "median_frac_pred_at_clip_bounds": float(
                group["frac_pred_at_clip_bounds"].median()
            ),
        }
        for name in E5_METRICS:
            values = group.loc[~group["fit_failed"], name].dropna()
            row[f"{name}_n_valid"] = len(values)
            difference_column = f"{name}_difference_vs_none"
            differences = (
                group.loc[~group["fit_failed"], difference_column].dropna()
                if difference_column in group.columns
                else pd.Series(dtype=float)
            )
            row[f"{difference_column}_n_valid"] = len(differences)
            for label, quantile in (("median", 0.5), ("q1", 0.25), ("q3", 0.75)):
                row[f"{name}_{label}"] = (
                    float(values.quantile(quantile)) if len(values) else np.nan
                )
                row[f"{difference_column}_{label}"] = (
                    float(differences.quantile(quantile))
                    if len(differences) else np.nan
                )
        summary_rows.append(row)
    return draws, pd.DataFrame(summary_rows)


def e5_seed_record(seed: int) -> str:
    return json.dumps(
        {"seed": seed, "rng": "numpy default_rng([seed, k])", "n_draws": E5_N_DRAWS,
         "budgets": list(E5_LABEL_BUDGETS)},
        sort_keys=True,
    )
