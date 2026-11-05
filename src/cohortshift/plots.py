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
    """Plot target reliability bins as annotated points."""
    if "outcome" in reliability_data.columns:
        reliability_data = reliability_data.loc[reliability_data["outcome"] == "NPM1"]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    for method_index, (method, rows) in enumerate(
        reliability_data.groupby("method", sort=True)
    ):
        axis.scatter(
            rows["mean_predicted_probability"],
            rows["observed_fraction"],
            label=method,
        )
        for row in rows.itertuples(index=False):
            axis.annotate(
                str(int(row.bin_count)),
                (row.mean_predicted_probability, row.observed_fraction),
                xytext=(0, 5 + method_index * 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
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
        & coverage_data["status"].eq("complete")
    ].dropna(subset=["target_coverage"])
    methods = selected["method"].tolist()
    positions = np.arange(len(selected))
    figure, (overall_axis, class_axis) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True
    )

    def interval_data(metric: str) -> tuple[np.ndarray, np.ndarray]:
        lower = selected[f"target_{metric}_ci_lower"].to_numpy(dtype=float)
        upper = selected[f"target_{metric}_ci_upper"].to_numpy(dtype=float)
        center = (lower + upper) / 2
        return center, np.vstack((center - lower, upper - center))

    overall_axis.bar(
        positions,
        selected["target_coverage"],
    )
    interval_center, interval_errors = interval_data("coverage")
    overall_axis.errorbar(
        positions,
        interval_center,
        yerr=interval_errors,
        fmt="none",
        ecolor="black",
        capsize=3,
        label="95% percentile CI",
    )
    overall_axis.axhline(0.9, linestyle="--", color="gray", label="Nominal 90%")
    overall_axis.set(
        ylabel="Coverage",
        ylim=(0, 1),
        title="Overall target split-conformal coverage",
    )
    overall_axis.legend()

    width = 0.36
    class_axis.bar(
        positions - width / 2,
        selected["target_positive_class_coverage"],
        width,
        label="Positive class",
    )
    positive_center, positive_errors = interval_data("positive_class_coverage")
    class_axis.errorbar(
        positions - width / 2,
        positive_center,
        yerr=positive_errors,
        fmt="none",
        ecolor="black",
        capsize=3,
        label="95% percentile CI",
    )
    class_axis.bar(
        positions + width / 2,
        selected["target_negative_class_coverage"],
        width,
        label="Negative class",
    )
    negative_center, negative_errors = interval_data("negative_class_coverage")
    class_axis.errorbar(
        positions + width / 2,
        negative_center,
        yerr=negative_errors,
        fmt="none",
        ecolor="black",
        capsize=3,
        label="95% percentile CI",
    )
    class_axis.axhline(0.9, linestyle="--", color="gray", label="Nominal 90%")
    class_axis.set(
        xlabel="Adaptation setting",
        ylabel="Coverage",
        ylim=(0, 1),
        title="Target coverage by true class",
        xticks=positions,
        xticklabels=methods,
    )
    class_axis.legend()
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
