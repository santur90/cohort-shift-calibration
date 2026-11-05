#!/usr/bin/env python3
"""Run the unsupervised E0 cohort-shift diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.shift import E0_METHODS, run_e0_shift  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"


def main(methods: tuple[str, ...]) -> int:
    source_path = PROCESSED_DIR / "aligned_beat_aml.tsv.gz"
    target_path = PROCESSED_DIR / "aligned_tcga_laml.tsv.gz"
    genes_path = PROCESSED_DIR / "source_train_top2000_genes.txt"
    source_splits_path = ROOT / "results" / "splits" / "beat_aml_splits.tsv"
    seeds_path = ROOT / "results" / "splits" / "seeds.json"
    required = (
        source_path,
        target_path,
        genes_path,
        source_splits_path,
        seeds_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared files: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            + ". Run `make prepare` first."
        )

    genes = genes_path.read_text(encoding="utf-8").splitlines()
    with seeds_path.open(encoding="utf-8") as handle:
        seed = int(json.load(handle)["e0_domain_cv_and_pca"])
    if len(genes) != len(set(genes)):
        raise ValueError("Source-train feature list contains duplicate genes")
    source = pd.read_csv(
        source_path,
        sep="\t",
        index_col="SAMPLE_ID",
        usecols=["SAMPLE_ID", *genes],
    ).loc[:, genes]
    target = pd.read_csv(
        target_path,
        sep="\t",
        index_col="SAMPLE_ID",
        usecols=["SAMPLE_ID", *genes],
    ).loc[:, genes]
    source_splits = pd.read_csv(
        source_splits_path,
        sep="\t",
        usecols=["sample_id", "split"],
        index_col="sample_id",
    )
    if source_splits.index.has_duplicates:
        raise ValueError("Beat AML split file contains duplicate sample IDs")
    source_train_samples = source_splits.index[source_splits["split"] == "train"]
    missing_train_samples = set(source_train_samples) - set(source.index)
    if not len(source_train_samples) or missing_train_samples:
        raise ValueError(
            "Beat AML training split is empty or references samples missing from "
            "the aligned source matrix"
        )
    source_train = source.loc[source_train_samples, genes]

    summary, gene_ks = run_e0_shift(
        source,
        target,
        methods,
        seed=seed,
        pca_output_dir=str(ROOT / "results" / "figures"),
        source_train=source_train,
    )
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "e0_shift.tsv", sep="\t", index=False)
    gene_ks.to_csv(tables / "e0_gene_ks.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {tables / 'e0_shift.tsv'}")
    print(f"Wrote per-gene KS statistics to {tables / 'e0_gene_ks.tsv'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=E0_METHODS,
        default=list(E0_METHODS),
        help="registered adaptation methods to evaluate in addition to none",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(tuple(dict.fromkeys(arguments.methods))))
