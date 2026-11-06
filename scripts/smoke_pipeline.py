#!/usr/bin/env python3
"""Exercise the analysis on a small, isolated, synthetic fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cohortshift import experiments
from cohortshift.plots import save_coverage_plot, save_target_reliability_plot
from cohortshift.recalibration import run_few_label_recalibration
from cohortshift.source_experiments import (
    SOURCE_EXPERIMENT_SEEDS,
    _apply_calibrator,
    run_source_experiments,
    save_source_reliability_plot,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smoke.json"


def table_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and (path.name.endswith(".tsv") or path.name.endswith(".tsv.gz"))
    }


def run_pipeline(output: Path) -> dict[str, str]:
    config = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    columns = [f"gene_{i:03d}" for i in range(config["genes"])]

    def cohort(n: int, prefix: str, shift: float) -> tuple[pd.DataFrame, pd.Series]:
        latent = rng.normal(size=(n, config["genes"]))
        labels = (latent[:, 0] + 0.4 * latent[:, 1] + rng.normal(size=n) > 0).astype(int)
        values = latent + shift
        if shift:
            values = np.exp2(values)
        ids = [
            f"synthetic_{i:04d}_BA{i:04d}" if prefix == "source"
            else f"TCGA-SM-{i:04d}-03"
            for i in range(n)
        ]
        return pd.DataFrame(values, index=ids, columns=columns), pd.Series(labels, index=ids)

    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir()
    figures.mkdir()
    raw = output / "raw"
    configs = []
    for name, count, prefix, shift in (
        ("beat_aml", config["source_samples"], "source", 0),
        ("tcga_laml", config["target_samples"], "target", config["target_shift"]),
    ):
        expression, labels = cohort(count, prefix, shift)
        study = raw / name
        study.mkdir(parents=True)
        matrix = expression.T
        matrix.index.name = "Hugo_Symbol"
        matrix.to_csv(study / "rna.txt", sep="\t", float_format="%.6f")
        clinical = pd.DataFrame({
            "SAMPLE_ID": expression.index,
            "COHORT": "synthetic",
            "ONCOTREE_CODE": "AML",
            "CANCER_TYPE_DETAILED": "AML",
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "IS_RELAPSE": "FALSE",
            "SPECIMEN_GROUPS": "Initial Acute Leukemia Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": 0,
        })
        clinical.to_csv(study / "data_clinical_sample.txt", sep="\t", index=False)
        pd.DataFrame({"PATIENT_ID": ["synthetic"], "SUBTYPE": ["AML"]}).to_csv(
            study / "data_clinical_patient.txt", sep="\t", index=False
        )
        pd.DataFrame({"SAMPLE_ID": expression.index, "mutations": "WXS"}).to_csv(
            study / "data_gene_panel_matrix.txt", sep="\t", index=False
        )
        mutations = [
            {"Hugo_Symbol": gene, "Tumor_Sample_Barcode": sample,
             "Variant_Classification": "Missense_Mutation"}
            for sample, label in labels.items() if label == 1
            for gene in ("NPM1", "DNMT3A")
        ]
        pd.DataFrame(mutations).to_csv(study / "data_mutations.txt", sep="\t", index=False)
        configs.append({"cohort": name, "study_id": name, "rna_file": "rna.txt"})
    spec = importlib.util.spec_from_file_location("smoke_prepare", ROOT / "scripts" / "01_prepare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load preparation script")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    prepare.ROOT = output
    prepare.RAW_DIR = raw
    prepare.PROCESSED_DIR = output / "processed"
    prepare.RESULTS_TABLES = tables
    prepare.SPLITS_DIR = output / "splits"
    with patch.object(prepare, "_read_config", return_value=configs):
        prepare.main("e6b")
    source = pd.read_csv(output / "processed" / "aligned_beat_aml_e6b.tsv.gz", sep="\t", index_col="SAMPLE_ID")
    target = pd.read_csv(output / "processed" / "aligned_tcga_laml_e6b.tsv.gz", sep="\t", index_col="SAMPLE_ID")
    source_y = pd.read_csv(output / "processed" / "labels_beat_aml_e6b.tsv", sep="\t", index_col="SAMPLE_ID")["NPM1"]
    target_y = pd.read_csv(output / "processed" / "labels_tcga_laml_e6b.tsv", sep="\t", index_col="SAMPLE_ID")["NPM1"].loc[target.index]
    patients = {sample: sample for sample in source.index}
    main_split = pd.read_csv(output / "splits" / "beat_aml_splits_e6b.tsv", sep="\t").set_index("sample_id")["split"].to_dict()
    source_results = run_source_experiments(
        source, source_y, patients, seeds=SOURCE_EXPERIMENT_SEEDS[:1],
        n_genes=config["selected_genes"], registered_main_split=main_split,
    )
    selected = source_results["selected_genes"]
    main_genes = selected.loc[selected["split_set"] == "registered_main_split", "gene"].tolist()
    repeat_genes = {1: selected.loc[selected["split_set"] == "20_repeats", "gene"].tolist()}
    bundle = experiments.prepare_cross_cohort_predictions(
        source, source_y, target, main_split, source_results["splits"],
        registered_main_genes=main_genes, repeat_genes=repeat_genes, seed=4201,
    )
    target_path = output / "target_labels.tsv"
    target_y.rename("NPM1").rename_axis("SAMPLE_ID").to_csv(target_path, sep="\t")
    originals = {
        name: getattr(experiments, name)
        for name in (
            "_bootstrap_metric_intervals", "_bootstrap_coverage_intervals",
            "_paired_brier_bootstrap_difference",
        )
    }
    with (
        patch.object(experiments, "_bootstrap_metric_intervals",
                     partial(originals["_bootstrap_metric_intervals"], n_resamples=10)),
        patch.object(experiments, "_bootstrap_coverage_intervals",
                     partial(originals["_bootstrap_coverage_intervals"], n_resamples=10)),
        patch.object(experiments, "_paired_brier_bootstrap_difference",
                     partial(originals["_paired_brier_bootstrap_difference"], n_resamples=10)),
    ):
        cross = experiments.final_evaluation(bundle, target_path, list(target.index), bootstrap_seed=4201)
    for name in ("e1", "e2", "reliability", "splits", "selected_genes"):
        source_results[name].to_csv(tables / f"source_{name}.tsv", sep="\t", index=False)
    for name, frame in cross.items():
        frame.to_csv(tables / f"cross_{name}.tsv", sep="\t", index=False)
    save_source_reliability_plot(source_results["reliability"], str(figures / "source.png"))
    save_target_reliability_plot(cross["reliability"], figures / "target.png")
    save_coverage_plot(cross["conformal"], figures / "coverage.png")
    for entry in bundle["main"]:
        if entry["method"] not in {"A0", "A0+A1"} or entry["failed"]:
            continue
        probabilities = _apply_calibrator(
            "none", None, entry["predictions"]["logistic"]["target_probabilities"]
        )
        draws, summary = run_few_label_recalibration(
            list(target.index), probabilities, target_y.to_numpy(),
            setting=entry["method"], seed=5301, budgets=(0, 10, 20, 40), n_draws=2,
        )
        draws.to_csv(tables / f"e5_{entry['method']}_draws.tsv", sep="\t", index=False)
        summary.to_csv(tables / f"e5_{entry['method']}_summary.tsv", sep="\t", index=False)
    hashes = table_hashes(output)
    if len(hashes) != 23:
        raise RuntimeError(f"Expected 23 output tables, found {len(hashes)}")
    if len(list(figures.glob("*.png"))) != 3:
        raise RuntimeError("Smoke pipeline did not produce all three plots")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reproduce and args.output:
        parser.error("--reproduce and --output cannot be combined")
    start = time.monotonic()
    if args.output:
        run_pipeline(args.output)
        return
    with tempfile.TemporaryDirectory(prefix="cohortshift-smoke-") as directory:
        if args.reproduce:
            for name in ("first", "second"):
                subprocess.run(
                    [sys.executable, __file__, "--output", str(Path(directory) / name)],
                    check=True,
                )
            first = table_hashes(Path(directory) / "first")
            second = table_hashes(Path(directory) / "second")
            if len(first) != 23 or len(second) != 23:
                raise RuntimeError("Both runs must produce all 23 output tables")
            if first != second:
                differences = sorted(name for name in first.keys() | second.keys() if first.get(name) != second.get(name))
                raise RuntimeError(f"Output table hashes differ: {differences}")
            print(f"SHA-256 identical for all {len(first)} output tables")
        else:
            first = run_pipeline(Path(directory) / "first")
            print(f"Synthetic pipeline produced {len(first)} tables and three figures")
    print(f"Elapsed: {time.monotonic() - start:.1f} seconds; PID {os.getpid()}")


if __name__ == "__main__":
    main()
