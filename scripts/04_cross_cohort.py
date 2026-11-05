#!/usr/bin/env python3
"""Run the preregistered E3/E4 cross-cohort and split-conformal evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.experiments import (  # noqa: E402
    final_evaluation,
    prepare_cross_cohort_predictions,
)
from cohortshift.plots import (  # noqa: E402
    save_coverage_plot,
    save_target_reliability_plot,
)

PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "results" / "splits"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
BOOTSTRAP_SEED = 4201


def _read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col="SAMPLE_ID")


def main() -> int:
    required = (
        PROCESSED / "aligned_beat_aml.tsv.gz",
        PROCESSED / "aligned_tcga_laml.tsv.gz",
        PROCESSED / "labels_beat_aml.tsv",
        PROCESSED / "labels_tcga_laml.tsv",
        PROCESSED / "source_train_top2000_genes.txt",
        SPLITS / "beat_aml_splits.tsv",
        SPLITS / "beat_aml_e1_e2_repeated_splits.tsv",
        SPLITS / "beat_aml_e1_e2_selected_genes.tsv",
        SPLITS / "tcga_laml_splits.tsv",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing E3/E4 inputs: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            + ". Run `make prepare`, then `make e1` first."
        )

    source_expression = _read_matrix(required[0])
    target_expression = _read_matrix(required[1])
    source_label_frame = pd.read_csv(
        required[2],
        sep="\t",
        usecols=["SAMPLE_ID", "NPM1", "DNMT3A"],
        dtype={"SAMPLE_ID": str},
    ).set_index("SAMPLE_ID")
    registered_main_split = pd.read_csv(
        required[5], sep="\t", dtype=str
    ).set_index("sample_id")["split"].to_dict()
    repeated_splits = pd.read_csv(required[6], sep="\t", dtype={"sample_id": str})
    selected_genes = pd.read_csv(required[7], sep="\t", dtype={"gene": str})
    repeat_genes = {
        int(repeat): group["gene"].tolist()
        for repeat, group in selected_genes.loc[
            selected_genes["split_set"] == "20_repeats"
        ].groupby("repeat", sort=True)
    }
    if len(repeat_genes) != 20 or any(len(genes) != 2000 for genes in repeat_genes.values()):
        raise ValueError("the saved E1/E2 repeated selections must contain 20 lists of 2,000 genes")
    main_gene_list = Path(
        PROCESSED / "source_train_top2000_genes.txt"
    ).read_text(encoding="utf-8").splitlines()
    if len(main_gene_list) != 2000 or len(set(main_gene_list)) != 2000:
        raise ValueError("registered main source-train gene list must have 2,000 genes")

    target_split = pd.read_csv(required[8], sep="\t", dtype=str)
    target_patient_ids = (
        target_split.set_index("sample_id")
        .loc[target_expression.index, "patient_id"]
        .astype(str)
        .tolist()
    )
    result_sets = []
    for outcome in ("NPM1", "DNMT3A"):
        source_labels = source_label_frame[outcome]
        profiled_source = source_labels.notna()
        outcome_expression = source_expression.loc[profiled_source]
        outcome_split = {
            sample: split
            for sample, split in registered_main_split.items()
            if sample in outcome_expression.index
        }
        outcome_repeats = repeated_splits.loc[
            repeated_splits["sample_id"].isin(outcome_expression.index)
        ].copy()
        prediction_bundle = prepare_cross_cohort_predictions(
            outcome_expression,
            source_labels.loc[profiled_source].astype(int),
            target_expression,
            outcome_split,
            outcome_repeats,
            registered_main_genes=main_gene_list,
            repeat_genes=repeat_genes,
            seed=BOOTSTRAP_SEED,
            outcome=outcome,
        )
        result_sets.append(
            final_evaluation(
                prediction_bundle,
                PROCESSED / "labels_tcga_laml.tsv",
                target_patient_ids,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
        )
    results = {
        key: pd.concat([result[key] for result in result_sets], ignore_index=True)
        for key in ("cross_cohort", "repeats", "conformal", "reliability")
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results["cross_cohort"].to_csv(
        TABLES / "e3_e4_cross_cohort.tsv", sep="\t", index=False
    )
    results["repeats"].to_csv(
        TABLES / "e3_e4_repeats.tsv", sep="\t", index=False
    )
    results["conformal"].to_csv(
        TABLES / "conformal_coverage.tsv", sep="\t", index=False
    )
    save_target_reliability_plot(
        results["reliability"], FIGURES / "reliability_target.png"
    )
    save_coverage_plot(results["conformal"], FIGURES / "coverage.png")
    print(results["cross_cohort"].to_string(index=False))
    print(f"Wrote {TABLES / 'e3_e4_cross_cohort.tsv'}")
    print(f"Wrote {TABLES / 'e3_e4_repeats.tsv'}")
    print(f"Wrote {TABLES / 'conformal_coverage.tsv'}")
    print(f"Wrote {FIGURES / 'reliability_target.png'}")
    print(f"Wrote {FIGURES / 'coverage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
