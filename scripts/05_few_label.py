#!/usr/bin/env python3
"""Run pre-registered E5 few-label target recalibration (exploratory)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.experiments import prepare_cross_cohort_predictions  # noqa: E402
from cohortshift.recalibration import (  # noqa: E402
    e5_seed_record,
    run_few_label_recalibration,
)
from cohortshift.source_experiments import _apply_calibrator  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "results" / "splits"
TABLES = ROOT / "results" / "tables"
E3_SEED = 4201  # same seed as E3 so frozen source-trained predictions match
E5_SEED = 5301
SETTINGS = (
    ("A0+A1", "primary", ""),
    (
        "A0",
        "secondary",
        "most predictions sit at the probability clip bounds, so "
        "recalibration is ill-posed in this setting",
    ),
)


def main() -> int:
    paths = {
        "source_x": PROCESSED / "aligned_beat_aml.tsv.gz",
        "target_x": PROCESSED / "aligned_tcga_laml.tsv.gz",
        "source_y": PROCESSED / "labels_beat_aml.tsv",
        "target_y": PROCESSED / "labels_tcga_laml.tsv",
        "genes": PROCESSED / "source_train_top2000_genes.txt",
        "split": SPLITS / "beat_aml_splits.tsv",
    }
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing E5 inputs: {', '.join(missing)}. Run `make prepare` first."
        )
    source_x = pd.read_csv(paths["source_x"], sep="\t", index_col="SAMPLE_ID")
    target_x = pd.read_csv(paths["target_x"], sep="\t", index_col="SAMPLE_ID")
    source_y = pd.read_csv(
        paths["source_y"], sep="\t", usecols=["SAMPLE_ID", "NPM1"],
        dtype={"SAMPLE_ID": str},
    ).set_index("SAMPLE_ID")["NPM1"]
    profiled = source_y.notna()
    source_x, source_y = source_x.loc[profiled[profiled].index], source_y[profiled]
    split = (
        pd.read_csv(paths["split"], sep="\t", dtype=str)
        .set_index("sample_id")["split"]
        .to_dict()
    )
    split = {k: v for k, v in split.items() if k in source_x.index}
    genes = paths["genes"].read_text(encoding="utf-8").splitlines()

    # Source-only training and calibration, identical to E3's main split.
    bundle = prepare_cross_cohort_predictions(
        source_x,
        source_y.astype(int),
        target_x,
        split,
        pd.DataFrame(columns=["repeat", "sample_id", "split", "split_set"]),
        registered_main_genes=genes,
        seed=E3_SEED,
        outcome="NPM1",
        methods=tuple(name for name, _, _ in SETTINGS),
        include_repeats=False,
    )
    # Target labels are first read here, after all predictions are frozen.
    target_labels = pd.read_csv(
        paths["target_y"], sep="\t", usecols=["SAMPLE_ID", "NPM1"],
        dtype={"SAMPLE_ID": str},
    ).set_index("SAMPLE_ID")["NPM1"].loc[bundle["target_samples"]]
    keep = target_labels.notna().to_numpy()
    sample_ids = [s for s, k in zip(bundle["target_samples"], keep) if k]
    y = target_labels[keep].to_numpy(dtype=int)

    draw_tables, summaries = [], []
    for entry in bundle["main"]:
        method = entry["method"]
        if entry["failed"]:
            raise RuntimeError(f"E5 setting {method} failed: {entry['failure_reason']}")
        _, role, note = next(item for item in SETTINGS if item[0] == method)
        raw = entry["predictions"]["logistic"]["target_probabilities"]
        probabilities = _apply_calibrator("none", None, raw)[keep]
        draws, summary = run_few_label_recalibration(
            sample_ids, probabilities, y, setting=method, seed=E5_SEED
        )
        summary.insert(1, "analysis_role", role)
        summary["note"] = note
        draw_tables.append(draws)
        summaries.append(summary)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        TABLES / "e5_few_label.tsv", sep="\t", index=False
    )
    pd.concat(draw_tables, ignore_index=True).to_csv(
        TABLES / "e5_few_label_draws.tsv", sep="\t", index=False
    )
    (SPLITS / "e5_seeds.json").write_text(e5_seed_record(E5_SEED) + "\n")
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))
    print(f"Wrote {TABLES / 'e5_few_label.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
