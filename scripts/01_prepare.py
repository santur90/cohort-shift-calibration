#!/usr/bin/env python3
"""Prepare registered AML analysis sets and patient-level splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.data import (  # noqa: E402
    detect_apl_samples,
    iter_tsv,
    load_expression_matrix,
    mutation_profile_panels,
    mutation_profile_samples,
    parse_patient_id,
    select_one_sample_per_patient,
    split_patient_ids,
    split_patient_ids_stratified,
)
from cohortshift.features import (  # noqa: E402
    filter_jointly_complete_genes,
    select_top_variance_genes,
)
from cohortshift.labels import labels_from_maf_rows  # noqa: E402

CONFIG = ROOT / "config.tsv"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_TABLES = ROOT / "results" / "tables"
SPLITS_DIR = ROOT / "results" / "splits"
SEEDS = {"beat_aml": 3101, "tcga_laml": 3102}
E0_SEED = 2026
GENES = {"NPM1", "DNMT3A"}
UNPROFILED_VALUES = {"", "NA", "N/A", "NULL", "NONE"}

# Beat AML = log2 RPKM, TCGA = RSEM (batch-corrected); the unit difference is part of the shift and must be reported.


def _read_config() -> list[dict[str, str]]:
    with CONFIG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _print_distinct(
    cohort: str,
    clinical: list[dict[str, str]],
    patient_clinical: list[dict[str, str]],
) -> None:
    if cohort == "beat_aml":
        fields = (
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION",
            "COHORT",
            "ONCOTREE_CODE",
            "CANCER_TYPE_DETAILED",
            "IS_RELAPSE",
            "SPECIMEN_GROUPS",
        )
        field_sources = {field: clinical for field in fields}
    else:
        field_sources = {
            "ONCOTREE_CODE": clinical,
            "SUBTYPE": patient_clinical,
        }
    for field, records in field_sources.items():
        if not records or field not in records[0]:
            raise ValueError(f"{cohort} clinical sample file lacks {field}")
        distinct = sorted({row.get(field, "") or "<blank>" for row in records})
        print(f"DATA CHECK {cohort} {field}: {' | '.join(distinct)}")


def _print_beat_cross_check(
    candidate_samples: set[str], clinical_by_sample: dict[str, dict[str, str]]
) -> None:
    combinations = sorted(
        {
            (
                clinical_by_sample[sample].get(
                    "DISEASE_STAGE_AT_SPECIMEN_COLLECTION", ""
                ),
                clinical_by_sample[sample].get("IS_RELAPSE", ""),
                clinical_by_sample[sample].get("SPECIMEN_GROUPS", ""),
            )
            for sample in candidate_samples
        }
    )
    print("DATA CHECK Beat AML stage / IS_RELAPSE / SPECIMEN_GROUPS:")
    for stage, relapse, groups in combinations:
        print("  " + " | ".join(value or "<blank>" for value in (stage, relapse, groups)))
    conflicts = []
    for sample in candidate_samples:
        row = clinical_by_sample[sample]
        if row.get("DISEASE_STAGE_AT_SPECIMEN_COLLECTION") != "Initial Diagnosis":
            continue
        relapse_flag = row.get("IS_RELAPSE", "").strip().casefold() in {
            "true",
            "yes",
            "1",
        }
        relapse_group = "relapse" in row.get("SPECIMEN_GROUPS", "").casefold()
        if relapse_flag or relapse_group:
            conflicts.append(sample)
    print(
        f"DATA CHECK Beat AML initial-diagnosis candidates flagged as relapse: "
        f"{len(conflicts)}"
    )
    for sample in conflicts:
        row = clinical_by_sample[sample]
        print(
            f"  review {sample}: IS_RELAPSE={row.get('IS_RELAPSE', '<blank>')}; "
            f"SPECIMEN_GROUPS={row.get('SPECIMEN_GROUPS', '<blank>')}"
        )


def _positive_count(labels: pd.DataFrame, gene: str) -> int:
    return int(labels[gene].eq(1).fillna(False).sum())


def _profiled_count(labels: pd.DataFrame, gene: str) -> int:
    return int(labels[gene].notna().sum())


def _output_path(directory: Path, filename: str, analysis_set: str) -> Path:
    if analysis_set not in {"main", "e6a", "e6b"}:
        raise ValueError(f"Unknown analysis set: {analysis_set}")
    if Path(filename).name != filename:
        raise ValueError("output filename must not contain a directory")
    suffix = "" if analysis_set == "main" else f"_{analysis_set}"
    stem, dot, extension = filename.partition(".")
    if not dot:
        raise ValueError("output filename must have an extension")
    return directory / f"{stem}{suffix}.{extension}"


def main(analysis_set: str = "main") -> int:
    if analysis_set not in {"main", "e6a", "e6b"}:
        raise ValueError(f"Unknown analysis set: {analysis_set}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cohort_data: dict[str, dict[str, object]] = {}

    for config in _read_config():
        cohort = config["cohort"]
        study_dir = RAW_DIR / config["study_id"]
        expression_path = study_dir / config["rna_file"]
        clinical_path = study_dir / "data_clinical_sample.txt"
        mutations_path = study_dir / "data_mutations.txt"
        panel_path = study_dir / "data_gene_panel_matrix.txt"
        patient_path = study_dir / "data_clinical_patient.txt"
        structural_variant_path = study_dir / "data_sv.txt"
        missing = [
            str(path.relative_to(ROOT))
            for path in (
                expression_path,
                clinical_path,
                patient_path,
                mutations_path,
                panel_path,
            )
            if not path.is_file()
        ]
        if analysis_set == "e6a" and not structural_variant_path.is_file():
            missing.append(str(structural_variant_path.relative_to(ROOT)))
        if missing:
            raise FileNotFoundError(
                f"Missing cohort inputs: {', '.join(missing)}. Run `make data`."
            )

        expression = load_expression_matrix(expression_path)
        print(f"INPUT RNA {cohort}: samples={expression.shape[0]}")
        clinical = list(iter_tsv(clinical_path))
        patient_clinical = list(iter_tsv(patient_path))
        if not clinical:
            raise ValueError(f"No clinical sample records found in {clinical_path}")
        sample_id_col = next(
            (column for column in clinical[0] if column.upper() == "SAMPLE_ID"), None
        )
        if sample_id_col is None:
            raise ValueError(f"{clinical_path} has no SAMPLE_ID column")
        clinical_by_sample = {
            row[sample_id_col]: row for row in clinical if row.get(sample_id_col)
        }

        panel_values = mutation_profile_panels(panel_path)
        mutation_profiled = {
            sample
            for sample, panel in panel_values.items()
            if panel.strip().upper() not in UNPROFILED_VALUES
        }
        candidate_samples = (
            set(expression.index) & set(clinical_by_sample) & mutation_profiled
        )
        _print_distinct(cohort, clinical, patient_clinical)
        if cohort == "beat_aml":
            _print_beat_cross_check(candidate_samples, clinical_by_sample)
        apl_samples = (
            detect_apl_samples(
                clinical_by_sample,
                iter_tsv(structural_variant_path),
                cohort=cohort,
            )
            if analysis_set == "e6a"
            else None
        )
        selected_samples = select_one_sample_per_patient(
            cohort,
            candidate_samples,
            clinical_by_sample,
            analysis_set=analysis_set,
            apl_samples=apl_samples,
        )
        patient_by_sample = {
            sample: parse_patient_id(cohort, sample) for sample in selected_samples
        }
        if len(set(patient_by_sample.values())) != len(selected_samples):
            raise RuntimeError(f"{cohort} analysis set contains multiple samples per patient")

        gene_profiled, _ = mutation_profile_samples(panel_path, GENES)
        mutation_rows = list(iter_tsv(mutations_path))
        labels = labels_from_maf_rows(
            mutation_rows,
            GENES,
            selected_samples,
            gene_profiled,
        ).reindex(sorted(selected_samples))
        cohort_data[cohort] = {
            "expression": expression.loc[sorted(selected_samples)],
            "labels": labels,
            "patients": patient_by_sample,
            "candidate_count": len(candidate_samples),
            "rna_matrix_samples": expression.shape[0],
        }

    source_expression, target_expression, dropped_genes = (
        filter_jointly_complete_genes(
            cohort_data["beat_aml"]["expression"],
            cohort_data["tcga_laml"]["expression"],
        )
    )
    if source_expression.shape[1] == 0:
        raise ValueError("No genes are completely measured in both cohorts")
    cohort_data["beat_aml"]["expression"] = source_expression
    cohort_data["tcga_laml"]["expression"] = target_expression
    shared_genes = source_expression.columns.tolist()
    if analysis_set == "main":
        with (
            _output_path(PROCESSED_DIR, "jointly_complete_gene_names.txt", analysis_set)
        ).open("w", encoding="utf-8") as handle:
            handle.write("\n".join(shared_genes) + "\n")
    print(
        "GENE AVAILABILITY dropped genes: "
        f"Beat AML={dropped_genes['source']}, "
        f"TCGA-LAML={dropped_genes['target']}, "
        f"total unique={dropped_genes['total']}"
    )
    print(
        "GENE AVAILABILITY not present in both cohorts: "
        f"Beat AML only={dropped_genes['source_only']}, "
        f"TCGA-LAML only={dropped_genes['target_only']}, "
        f"total={dropped_genes['not_shared_total']}"
    )
    print(
        "GENE AVAILABILITY shared but incomplete: "
        f"Beat AML missing={dropped_genes['missing_source']}, "
        f"TCGA-LAML missing={dropped_genes['missing_target']}, "
        f"unique incomplete={dropped_genes['missing_total']}"
    )
    print(f"GENE AVAILABILITY kept in both cohorts={dropped_genes['kept']}")
    if analysis_set == "main" and dropped_genes["kept"] != 14005:
        raise ValueError(
            "Expected 14,005 genes measured in every sample of both main "
            f"analysis sets; found {dropped_genes['kept']}"
        )
    for cohort, values in cohort_data.items():
        expression = values["expression"]
        values["expression"] = expression
        expression.to_csv(
            _output_path(PROCESSED_DIR, f"aligned_{cohort}.tsv.gz", analysis_set),
            sep="\t",
            index_label="SAMPLE_ID",
            compression={"method": "gzip", "mtime": 0},
        )
        values["labels"].to_csv(
            _output_path(PROCESSED_DIR, f"labels_{cohort}.tsv", analysis_set),
            sep="\t",
            index_label="SAMPLE_ID",
        )

    analysis_rows = []
    split_rows: dict[str, list[dict[str, str]]] = {}
    dnmt3a_split_rows: list[dict[str, str]] = []
    beat_aml_split_counts: list[dict[str, str]] = []
    seed_info = {
        "analysis_set": analysis_set,
        "strategy": {
            "beat_aml_main": "patient-level 60/20/20 stratified by NPM1",
            "beat_aml_e7": "patient-level 60/20/20 stratified by DNMT3A",
            "tcga_laml": "patient-level shuffled 60/20/20 without target-label stratification",
        },
        "label_stratification": {
            "beat_aml_main": "NPM1",
            "beat_aml_e7": "DNMT3A",
            "tcga_laml": None,
        },
        "seeds": {
            **SEEDS,
            "beat_aml_e7_dnmt3a": SEEDS["beat_aml"] + 1,
        },
        "e0_domain_cv_and_pca": E0_SEED,
    }
    for cohort, values in cohort_data.items():
        patients = values["patients"]
        labels = values["labels"]
        if cohort == "beat_aml":
            npm1_labels = {
                patients[sample]: None if pd.isna(label) else int(label)
                for sample, label in labels["NPM1"].items()
            }
            patient_splits = split_patient_ids_stratified(
                npm1_labels, seed=SEEDS[cohort]
            )
            dnmt3a_labels = {
                patients[sample]: None if pd.isna(label) else int(label)
                for sample, label in labels["DNMT3A"].items()
            }
            dnmt3a_patient_splits = split_patient_ids_stratified(
                dnmt3a_labels, seed=SEEDS[cohort] + 1
            )
            dnmt3a_split_rows = [
                {
                    "cohort": cohort,
                    "sample_id": sample,
                    "patient_id": patient,
                    "split": dnmt3a_patient_splits[patient],
                }
                for sample, patient in sorted(patients.items())
            ]
        else:
            patient_splits = split_patient_ids(
                set(patients.values()), seed=SEEDS[cohort]
            )
        rows = [
            {
                "cohort": cohort,
                "sample_id": sample,
                "patient_id": patient,
                "split": patient_splits[patient],
            }
            for sample, patient in sorted(patients.items())
        ]
        split_rows[cohort] = rows

        if cohort == "beat_aml":
            for label_name, assigned_rows, gene in (
                ("main", rows, "NPM1"),
                ("e7_dnmt3a", dnmt3a_split_rows, "DNMT3A"),
            ):
                label_splits = {
                    row["sample_id"]: row["split"] for row in assigned_rows
                }
                for split in ("train", "validation", "test"):
                    samples = {
                        sample
                        for sample, assigned_split in label_splits.items()
                        if assigned_split == split
                    }
                    split_labels = labels.loc[sorted(samples), gene]
                    split_counts = {
                        "analysis": label_name,
                        "label": gene,
                        "split": split,
                        "n_samples": str(len(samples)),
                        "profiled_samples": str(int(split_labels.notna().sum())),
                        "positives": str(
                            int(split_labels.eq(1).fillna(False).sum())
                        ),
                    }
                    beat_aml_split_counts.append(split_counts)
                    print(
                        f"SPLIT beat_aml {label_name} {gene} {split}: "
                        f"N={split_counts['n_samples']}, "
                        f"positives={split_counts['positives']}/"
                        f"{split_counts['profiled_samples']} profiled"
                    )
        row = {
            "cohort": cohort,
            "rna_matrix_samples": str(values["rna_matrix_samples"]),
            "rna_and_mutation_profiled_candidates": str(values["candidate_count"]),
            "n_samples": str(len(patients)),
            "n_patients": str(len(set(patients.values()))),
            "npm1_profiled_samples": str(_profiled_count(labels, "NPM1")),
            "npm1_positives": str(_positive_count(labels, "NPM1")),
            "dnmt3a_profiled_samples": str(_profiled_count(labels, "DNMT3A")),
            "dnmt3a_positives": str(_positive_count(labels, "DNMT3A")),
        }
        analysis_rows.append(row)
        print(
            f"FINAL {cohort}: N={row['n_samples']}, patients={row['n_patients']}, "
            f"NPM1+={row['npm1_positives']}/{row['npm1_profiled_samples']} profiled, "
            f"DNMT3A+={row['dnmt3a_positives']}/{row['dnmt3a_profiled_samples']} profiled"
        )

    for cohort, rows in split_rows.items():
        with _output_path(SPLITS_DIR, f"{cohort}_splits.tsv", analysis_set).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("cohort", "sample_id", "patient_id", "split"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
    with (
        _output_path(SPLITS_DIR, "beat_aml_splits_dnmt3a_e7.tsv", analysis_set)
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cohort", "sample_id", "patient_id", "split"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(dnmt3a_split_rows)
    with _output_path(SPLITS_DIR, "seeds.json", analysis_set).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(seed_info, handle, indent=2)
        handle.write("\n")

    with _output_path(SPLITS_DIR, "beat_aml_split_counts.tsv", analysis_set).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "analysis",
            "label",
            "split",
            "n_samples",
            "profiled_samples",
            "positives",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(beat_aml_split_counts)

    analysis_path = _output_path(RESULTS_TABLES, "analysis_set.tsv", analysis_set)
    with analysis_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "cohort",
            "rna_matrix_samples",
            "rna_and_mutation_profiled_candidates",
            "n_samples",
            "n_patients",
            "npm1_profiled_samples",
            "npm1_positives",
            "dnmt3a_profiled_samples",
            "dnmt3a_positives",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(analysis_rows)

    source = cohort_data["beat_aml"]
    source_splits = {
        row["sample_id"]: row["split"] for row in split_rows["beat_aml"]
    }
    source_train = source["expression"].loc[
        [sample for sample, split in source_splits.items() if split == "train"]
    ]
    selected_genes = select_top_variance_genes(source_train, n_genes=2000)
    if len(selected_genes) != 2000:
        raise ValueError(
            f"Expected 2,000 jointly complete genes, found {len(selected_genes)}"
        )
    assert not source["expression"].loc[:, selected_genes].isna().any().any()
    assert not cohort_data["tcga_laml"]["expression"].loc[
        :, selected_genes
    ].isna().any().any()
    with _output_path(PROCESSED_DIR, "source_train_top2000_genes.txt", analysis_set).open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(selected_genes) + "\n")
    print(f"Aligned shared genes: {len(shared_genes)}")
    print(
        f"Selected top-variance source-train genes: {len(selected_genes)} "
        f"from {len(source_train)} Beat AML training samples"
    )
    for cohort, values in (
        ("Beat AML", source["expression"].loc[:, selected_genes].to_numpy(dtype=float)),
        (
            "TCGA-LAML",
            cohort_data["tcga_laml"]["expression"]
            .loc[:, selected_genes]
            .to_numpy(dtype=float),
        ),
    ):
        flattened = values.ravel()
        percentiles = np.percentile(flattened, [1, 50, 99])
        print(
            f"EXPRESSION {cohort} selected-2000 values: "
            f"min={flattened.min():.6g}; "
            f"p01={percentiles[0]:.6g}; "
            f"p50={percentiles[1]:.6g}; "
            f"p99={percentiles[2]:.6g}; "
            f"max={flattened.max():.6g}; "
            f"fraction_exact_zero={np.equal(flattened, 0).mean():.6g}; "
            f"fraction_negative={np.less(flattened, 0).mean():.6g}"
        )
    print(
        f"Wrote {analysis_path.relative_to(ROOT)} and split assignments "
        f"for analysis set {analysis_set}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-set",
        choices=("main", "e6a", "e6b"),
        default="main",
    )
    raise SystemExit(main(parser.parse_args().analysis_set))
