"""Plotting helpers for calibration results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_calibration_plot(
    labels: np.ndarray, probabilities: np.ndarray, path: str | Path
) -> None:
    """Save a reliability diagram with equal-width probability bins."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    observed = np.full(10, np.nan)
    for index in range(10):
        mask = (p >= bins[index]) & (
            p < bins[index + 1] if index < 9 else p <= bins[index + 1]
        )
        if mask.any():
            observed[index] = y[mask].mean()
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axis.plot(centers, observed, marker="o", label="Observed")
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_target_reliability_plot(
    reliability_data: pd.DataFrame, path: str | Path
) -> None:
    """Plot target reliability curves for the primary registered model."""
    if "outcome" in reliability_data.columns:
        reliability_data = reliability_data.loc[reliability_data["outcome"] == "NPM1"]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    for method, rows in reliability_data.groupby("method", sort=True):
        axis.plot(
            rows["mean_predicted_probability"],
            rows["observed_fraction"],
            marker="o",
            label=method,
        )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed fraction",
        title="Target reliability (registered main split; logistic, no calibration)",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_coverage_plot(coverage_data: pd.DataFrame, path: str | Path) -> None:
    """Plot target split-conformal coverage for the registered main split."""
    selected = coverage_data.loc[
        (coverage_data["split_set"] == "registered_main_split")
        & (coverage_data["model"] == "logistic")
        & (coverage_data["calibrator"] == "none")
        & (coverage_data["outcome"] == "NPM1")
    ]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(selected["method"], selected["target_coverage"])
    axis.axhline(0.9, linestyle="--", color="gray", label="Nominal 90%")
    axis.set(
        xlabel="Adaptation setting",
        ylabel="Target coverage",
        ylim=(0, 1),
        title="Split-conformal target coverage",
    )
    axis.legend()
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
