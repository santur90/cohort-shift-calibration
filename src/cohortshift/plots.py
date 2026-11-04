"""Plotting helpers for calibration results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
