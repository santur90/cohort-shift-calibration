#!/usr/bin/env python3
"""Run the preregistered E3/E4 cross-cohort and split-conformal evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.experiments import (  # noqa: E402
    add_e2_source_reference,
    final_evaluation,
    prepare_cross_cohort_predictions,
)
from cohortshift.plots import (  # noqa: E402
    save_coverage_plot,
    save_target_reliability_plot,
)
from cohortshift.source_experiments import (  # noqa: E402
    SOURCE_EXPERIMENT_SEEDS,
    repeated_split_table,
)

PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "results" / "splits"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
BOOTSTRAP_SEED = 4201


def _read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col="SAMPLE_ID")


def _print_set_counts(
    analysis_set: str,
    source_labels: pd.DataFrame,
    source_splits: dict[str, str],
    target_labels: pd.DataFrame,
) -> None:
    """Print N and NPM1 positives per cohort and source split before any model."""
    for split in ("train", "validation", "test"):
        samples = [s for s, v in source_splits.items() if v == split]
        npm1 = source_labels.loc[samples, "NPM1"]
        print(
            f"COUNTS {analysis_set} beat_aml {split}: N={len(samples)}, "
            f"NPM1+={int(npm1.eq(1).sum())}"
        )
    print(
        f"COUNTS {analysis_set} beat_aml all: N={len(source_labels)}, "
        f"NPM1+={int(source_labels['NPM1'].eq(1).sum())}"
    )
    print(
        f"COUNTS {analysis_set} tcga_laml all (evaluation set, not split): "
        f"N={len(target_labels)}, NPM1+={int(target_labels['NPM1'].eq(1).sum())}"
    )


def main(analysis_set: str = "main") -> int:
    suffix = "" if analysis_set == "main" else f"_{analysis_set}"
    required = (
        PROCESSED / f"aligned_beat_aml{suffix}.tsv.gz",
        PROCESSED / f"aligned_tcga_laml{suffix}.tsv.gz",
        PROCESSED / f"labels_beat_aml{suffix}.tsv",
        PROCESSED / f"labels_tcga_laml{suffix}.tsv",
        PROCESSED / f"source_train_top2000_genes{suffix}.txt",
        SPLITS / f"beat_aml_splits{suffix}.tsv",
        SPLITS / "beat_aml_e1_e2_repeated_splits.tsv",
        SPLITS / "beat_aml_e1_e2_selected_genes.tsv",
        SPLITS / f"tcga_laml_splits{suffix}.tsv",
        TABLES / "e2_calibration.tsv",
    )
    optional = () if analysis_set == "main" else (required[6], required[7], required[9])
    missing = [
        path for path in required if not path.is_file() and path not in optional
    ]
    if missing:
        raise FileNotFoundError(
            "Missing E3/E4 inputs: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            + ". Run `make prepare` (and `make e1` for the main set) first."
        )

    source_expression = _read_matrix(required[0])
    target_expression = _read_matrix(required[1])
    if analysis_set == "main" and len(target_expression) != 173:
        raise ValueError(
            "registered TCGA-LAML target expression must contain 173 samples; "
            f"found {len(target_expression)}"
        )
    source_label_frame = pd.read_csv(
        required[2],
        sep="\t",
        usecols=["SAMPLE_ID", "NPM1", "DNMT3A"],
        dtype={"SAMPLE_ID": str},
    ).set_index("SAMPLE_ID")
    registered_main_split = pd.read_csv(
        required[5], sep="\t", dtype=str
    ).set_index("sample_id")["split"].to_dict()
    main_split_counts = pd.Series(registered_main_split).value_counts().to_dict()
    if analysis_set == "main" and main_split_counts != {
        "train": 256,
        "validation": 85,
        "test": 85,
    }:
        raise ValueError(
            "registered Beat AML split must be 256/85/85; "
            f"found {main_split_counts}"
        )
    _print_set_counts(
        analysis_set,
        source_label_frame,
        registered_main_split,
        pd.read_csv(
            required[3], sep="\t", usecols=["SAMPLE_ID", "NPM1"]
        ),
    )
    if analysis_set == "main":
        repeated_splits = pd.read_csv(
            required[6], sep="\t", dtype={"sample_id": str}
        )
        selected_genes = pd.read_csv(required[7], sep="\t", dtype={"gene": str})
        repeat_genes = {
            int(repeat): group["gene"].tolist()
            for repeat, group in selected_genes.loc[
                selected_genes["split_set"] == "20_repeats"
            ].groupby("repeat", sort=True)
        }
        if len(repeat_genes) != 20 or any(
            len(genes) != 2000 or len(set(genes)) != 2000
            for genes in repeat_genes.values()
        ):
            raise ValueError(
                "the saved E1/E2 repeated selections must contain 20 lists of "
                "2,000 unique genes"
            )
    else:
        # Regenerate repeated splits for the new source set; each repeat's
        # top-2000 genes are selected on its own TRAIN split inside E3/E4.
        patient_by_sample = (
            pd.read_csv(required[5], sep="\t", dtype=str)
            .set_index("sample_id")["patient_id"]
            .to_dict()
        )
        repeated_splits = repeated_split_table(
            source_label_frame["NPM1"],
            patient_by_sample,
            seeds=SOURCE_EXPERIMENT_SEEDS,
        )
        repeated_splits.to_csv(
            SPLITS / f"beat_aml_e3_repeated_splits{suffix}.tsv",
            sep="\t",
            index=False,
        )
        repeat_genes = None
    main_gene_list = Path(required[4]).read_text(encoding="utf-8").splitlines()
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
                required[3],
                target_patient_ids,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
        )
    results = {
        key: pd.concat([result[key] for result in result_sets], ignore_index=True)
        for key in ("cross_cohort", "repeats", "conformal", "reliability")
    }
    if analysis_set == "main":
        results["cross_cohort"] = add_e2_source_reference(
            results["cross_cohort"], pd.read_csv(required[9], sep="\t"), repeated_splits
        )
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results["cross_cohort"].to_csv(
        TABLES / f"e3_e4_cross_cohort{suffix}.tsv", sep="\t", index=False
    )
    results["repeats"].to_csv(
        TABLES / f"e3_e4_repeats{suffix}.tsv", sep="\t", index=False
    )
    results["conformal"].to_csv(
        TABLES / f"conformal_coverage{suffix}.tsv", sep="\t", index=False
    )
    save_target_reliability_plot(
        results["reliability"], FIGURES / f"reliability_target{suffix}.png"
    )
    save_coverage_plot(results["conformal"], FIGURES / f"coverage{suffix}.png")
    print(results["cross_cohort"].to_string(index=False))
    print(f"Wrote {TABLES / f'e3_e4_cross_cohort{suffix}.tsv'}")
    print(f"Wrote {TABLES / f'e3_e4_repeats{suffix}.tsv'}")
    print(f"Wrote {TABLES / f'conformal_coverage{suffix}.tsv'}")
    print(f"Wrote {FIGURES / f'reliability_target{suffix}.png'}")
    print(f"Wrote {FIGURES / f'coverage{suffix}.png'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-set", choices=("main", "e6a", "e6b"), default="main"
    )
    raise SystemExit(main(parser.parse_args().analysis_set))
