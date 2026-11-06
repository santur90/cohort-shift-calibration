import importlib.util
from pathlib import Path

import pandas as pd


def load_report():
    path = Path(__file__).resolve().parents[1] / "scripts" / "update_results.py"
    spec = importlib.util.spec_from_file_location("update_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_results_are_not_fabricated(tmp_path):
    text = load_report().report(tmp_path)
    assert "E7 has not been reported" in text
    assert text.count("Results unavailable.") >= 5


def test_e5_results_are_taken_from_table_and_preserve_missing_values(tmp_path):
    row = {
        "setting": "A0+A1", "k": 10, "method": "Platt", "fraction_fit_failed": 0.125,
    }
    for metric, value in (
        ("brier", 0.1234), ("calibration_intercept", 0.2),
        ("calibration_slope", None),
    ):
        for stem in (metric, f"{metric}_difference_vs_none"):
            for suffix in ("median", "q1", "q3"):
                row[f"{stem}_{suffix}"] = value
    pd.DataFrame([row]).to_csv(tmp_path / "e5_few_label.tsv", sep="\t", index=False)
    text = load_report().report(tmp_path)
    assert "0.123 [0.123, 0.123]" in text
    assert "NA [NA, NA]" in text
    assert "0.125" in text
