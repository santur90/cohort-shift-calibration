#!/usr/bin/env python3
"""Run source-only E1 baseline and E2 calibration experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.source_experiments import (  # noqa: E402
    SOURCE_EXPERIMENT_SEEDS,
    SOURCE_MAIN_SPLIT_SEED,
    run_source_experiments,
    save_source_reliability_plot,
)

PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "results" / "splits"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"


def main(experiment: str) -> int:
    expression_path = PROCESSED_DIR / "aligned_beat_aml.tsv.gz"
    labels_path = PROCESSED_DIR / "labels_beat_aml.tsv"
    genes_path = PROCESSED_DIR / "jointly_complete_gene_names.txt"
    patient_list_path = SPLITS_DIR / "beat_aml_splits.tsv"
    required = (expression_path, labels_path, genes_path, patient_list_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source-only experiment inputs: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            + ". Run `make prepare` first."
        )

    genes = genes_path.read_text(encoding="utf-8").splitlines()
    if len(genes) != 14005 or len(genes) != len(set(genes)):
        raise ValueError(
            "The saved jointly complete source gene list must contain exactly "
            "14,005 unique names"
        )
    saved_patients = pd.read_csv(
        patient_list_path,
        sep="\t",
        usecols=["cohort", "sample_id", "patient_id", "split"],
        dtype=str,
    )
    if set(saved_patients["cohort"]) != {"beat_aml"}:
        raise ValueError("The saved patient list must contain Beat AML only")
    if len(saved_patients) != 426 or saved_patients["patient_id"].nunique() != 426:
        raise ValueError(
            "The saved Beat AML main patient list must contain 426 distinct patients"
        )
    patient_by_sample = saved_patients.set_index("sample_id")["patient_id"].to_dict()
    registered_main_split = saved_patients.set_index("sample_id")["split"].to_dict()
    split_counts = saved_patients["split"].value_counts().to_dict()
    expected_split_counts = {"train": 256, "validation": 85, "test": 85}
    if split_counts != expected_split_counts:
        raise ValueError(
            "The registered Beat AML main split must be 256/85/85; found "
            f"{split_counts}"
        )

    source_expression = pd.read_csv(
        expression_path,
        sep="\t",
        index_col="SAMPLE_ID",
        usecols=["SAMPLE_ID", *genes],
    ).loc[:, genes]
    source_labels_frame = pd.read_csv(
        labels_path,
        sep="\t",
        usecols=["SAMPLE_ID", "NPM1"],
        dtype={"SAMPLE_ID": str},
    ).set_index("SAMPLE_ID")
    source_labels = source_labels_frame["NPM1"]
    if set(source_expression.index) != set(patient_by_sample):
        raise ValueError(
            "The Beat AML expression rows do not match the saved main patient list"
        )

    results = run_source_experiments(
        source_expression,
        source_labels,
        patient_by_sample,
        seeds=SOURCE_EXPERIMENT_SEEDS,
        registered_main_split=registered_main_split,
    )
    _print_hyperparameter_distribution(results["e1"])
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "e1_e2_seeds.json").write_text(
        json.dumps(
            {
                "repeats": len(SOURCE_EXPERIMENT_SEEDS),
                "seeds": list(SOURCE_EXPERIMENT_SEEDS),
                "registered_main_split_seed": SOURCE_MAIN_SPLIT_SEED,
                "registered_main_split_is_one_of_20_repeats": False,
                "split": (
                    "patient-stratified 60/20/20; reserve 40% holdout, "
                    "then split holdout equally into calibration and test"
                ),
                "cv": "5-fold stratified on the training split only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results["splits"].to_csv(
        SPLITS_DIR / "beat_aml_e1_e2_repeated_splits.tsv",
        sep="\t",
        index=False,
    )
    results["selected_genes"].to_csv(
        SPLITS_DIR / "beat_aml_e1_e2_selected_genes.tsv",
        sep="\t",
        index=False,
    )

    if experiment in {"e1", "both"}:
        results["e1"].to_csv(
            TABLES_DIR / "e1_baseline.tsv", sep="\t", index=False
        )
        print(
            results["e1"].loc[
                results["e1"]["row_type"].isin(["mean", "sd"])
            ].to_string(index=False)
        )
        print(f"Wrote {TABLES_DIR / 'e1_baseline.tsv'}")
    if experiment in {"e2", "both"}:
        results["e2"].to_csv(
            TABLES_DIR / "e2_calibration.tsv", sep="\t", index=False
        )
        results["reliability"].to_csv(
            TABLES_DIR / "e2_reliability.tsv", sep="\t", index=False
        )
        save_source_reliability_plot(
            results["reliability"],
            str(FIGURES_DIR / "reliability_source.png"),
        )
        print(
            results["e2"].loc[
                results["e2"]["row_type"].isin(
                    ["mean", "sd", "median", "q1", "q3"]
                )
            ].to_string(index=False)
        )
        print(f"Wrote {TABLES_DIR / 'e2_calibration.tsv'}")
        print(f"Wrote {TABLES_DIR / 'e2_reliability.tsv'}")
        print(f"Wrote {FIGURES_DIR / 'reliability_source.png'}")
    return 0


def _print_hyperparameter_distribution(e1_table: pd.DataFrame) -> None:
    repeated = e1_table.loc[
        (e1_table["row_type"] == "repeat")
        & (e1_table["split_set"] == "20_repeats")
    ]
    logistic = repeated.loc[repeated["model"] == "logistic"]
    chosen_c = logistic["hyperparameters"].map(
        lambda value: json.loads(value)["classifier__C"]
    )
    c_counts = chosen_c.value_counts().sort_index()
    edge_count = int(chosen_c.isin({0.001, 1.0}).sum())
    print(f"CHOSEN logistic C distribution: {c_counts.to_dict()}")
    print(
        f"CHOSEN logistic C at grid edge: {edge_count}/{len(chosen_c)} repeats; "
        f"majority_flag={edge_count > len(chosen_c) / 2}"
    )
    hgb = repeated.loc[repeated["model"] == "hist_gradient_boosting"]
    chosen_hgb = hgb["hyperparameters"].map(
        lambda value: json.loads(value)
    )
    hgb_counts = pd.Series(
        [
            (
                parameters["max_depth"],
                parameters["learning_rate"],
            )
            for parameters in chosen_hgb
        ]
    ).value_counts().sort_index()
    print(
        "CHOSEN HGB max_depth/learning_rate distribution: "
        f"{hgb_counts.to_dict()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("e1", "e2", "both"),
        default="both",
    )
    raise SystemExit(main(parser.parse_args().experiment))
