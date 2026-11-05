#!/usr/bin/env python3
"""Summarize locally fetched cBioPortal AML inputs."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.data import (  # noqa: E402
    find_header,
    iter_tsv,
    matrix_sample_columns,
    mutation_profile_panels,
    mutation_profile_samples,
    select_one_sample_per_patient,
    sample_id_format,
)
from cohortshift.labels import (  # noqa: E402
    diagnosis_relapse_candidates,
    is_non_silent_variant,
)

CONFIG = ROOT / "config.tsv"
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables" / "inventory.tsv"
ANALYSIS_OUT = ROOT / "results" / "tables" / "analysis_set.tsv"

FILES_COMMON = (
    "data_clinical_sample.txt",
    "data_clinical_patient.txt",
    "data_mutations.txt",
    "data_gene_panel_matrix.txt",
)
BEAT_AML_DISTINCT_FIELDS = (
    "COHORT",
    "ONCOTREE_CODE",
    "CANCER_TYPE_DETAILED",
    "DISEASE_STAGE_AT_SPECIMEN_COLLECTION",
    "IS_RELAPSE",
    "SPECIMEN_GROUPS",
)
TCGA_PATIENT_DISTINCT_FIELDS = ("SUBTYPE",)
PATIENT_SUFFIXES = {
    "beat_aml": re.compile(r"_BA\d{4}$"),
    "tcga_laml": re.compile(r"-03$"),
}
DISPLAY_NAMES = {"beat_aml": "Beat AML", "tcga_laml": "TCGA-LAML"}
UNPROFILED_VALUES = {"", "NA", "N/A", "NULL", "NONE"}


def load_config() -> list[dict[str, str]]:
    with CONFIG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def clinical_values(path: Path) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {}
    for row in iter_tsv(path):
        for name, value in row.items():
            if value:
                columns.setdefault(name, []).append(value)
    return columns


def candidate_text(candidates: dict[str, list[str]]) -> str:
    return ";".join(
        f"{name}=[{','.join(values)}]" for name, values in candidates.items()
    ) or "none detected"


def find_column(headers: Iterable[str], *names: str) -> str:
    lookup = {name.casefold(): name for name in headers}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    raise ValueError(f"Required column not found; tried {', '.join(names)}")


def positive_samples(
    path: Path, genes: set[str], profiled_samples: dict[str, set[str]]
) -> dict[str, set[str]]:
    headers = find_header(path)
    gene_col = find_column(headers, "Hugo_Symbol", "Gene")
    sample_col = find_column(headers, "Tumor_Sample_Barcode", "SAMPLE_ID")
    variant_col = find_column(headers, "Variant_Classification")
    positives = {gene: set() for gene in genes}
    for row in iter_tsv(path):
        gene = row.get(gene_col, "").upper()
        sample = row.get(sample_col, "")
        if (
            gene in genes
            and sample in profiled_samples[gene]
            and is_non_silent_variant(row.get(variant_col, ""))
        ):
            positives[gene].add(sample)
    return positives


def patient_id(cohort_name: str, sample_id: str) -> str:
    suffix = PATIENT_SUFFIXES[cohort_name]
    patient, count = suffix.subn("", sample_id)
    if count != 1:
        raise ValueError(
            f"Sample ID {sample_id!r} does not match the registered "
            f"{cohort_name} patient-ID suffix"
        )
    return patient


def distinct_values(records: list[dict[str, str]], column: str) -> list[str]:
    values = {row.get(column, "") for row in records}
    return sorted("<blank>" if value == "" else value for value in values)


def print_data_check_values(
    cohort_name: str,
    records: list[dict[str, str]],
    fields: tuple[str, ...],
    rows: list[dict[str, str]],
    *,
    headers: list[str],
    file_name: str,
) -> None:
    header_set = set(headers)
    for column in fields:
        if column not in header_set:
            printable = "(column not present)"
        else:
            printable = " | ".join(distinct_values(records, column))
        print(f"DATA CHECK {DISPLAY_NAMES[cohort_name]} {column}: {printable}")
        rows.append(
            {
                "cohort": cohort_name,
                "record_type": "distinct_values",
                "file": file_name,
                "notes": f"{column}={printable}",
            }
        )


def select_beat_aml_samples(
    candidates: set[str], records_by_sample: dict[str, dict[str, str]]
) -> set[str]:
    fields = (
        "DISEASE_STAGE_AT_SPECIMEN_COLLECTION",
        "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION",
    )
    for field in fields:
        if records_by_sample and field not in next(iter(records_by_sample.values())):
            raise ValueError(f"Expected Beat AML clinical column {field!r}")

    cross_checks = sorted(
        {
            (
                records_by_sample[sample].get("DISEASE_STAGE_AT_SPECIMEN_COLLECTION", ""),
                records_by_sample[sample].get("IS_RELAPSE", ""),
                records_by_sample[sample].get("SPECIMEN_GROUPS", ""),
            )
            for sample in candidates
        }
    )
    print("DATA CHECK Beat AML eligible stage / IS_RELAPSE / SPECIMEN_GROUPS:")
    for values in cross_checks:
        print("  " + " | ".join(value or "<blank>" for value in values))
    initial_diagnosis_samples = [
        sample
        for sample in candidates
        if records_by_sample[sample][
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION"
        ]
        == "Initial Diagnosis"
    ]
    relapse_true_values = {"true", "yes", "1"}
    discordant_initial_samples = []
    for sample in initial_diagnosis_samples:
        row = records_by_sample[sample]
        relapse_flag = row.get("IS_RELAPSE", "").strip().casefold()
        relapse_group = "relapse" in row.get("SPECIMEN_GROUPS", "").casefold()
        if relapse_flag in relapse_true_values or relapse_group:
            discordant_initial_samples.append(sample)
    print(
        "DATA CHECK Beat AML initial-diagnosis candidates: "
        f"{len(initial_diagnosis_samples)}; flagged by IS_RELAPSE or "
        f"SPECIMEN_GROUPS: {len(discordant_initial_samples)}"
    )
    for sample in discordant_initial_samples:
        row = records_by_sample[sample]
        print(
            f"  review {sample}: IS_RELAPSE={row.get('IS_RELAPSE', '<blank>')}; "
            f"SPECIMEN_GROUPS={row.get('SPECIMEN_GROUPS', '<blank>')}"
        )

    return select_one_sample_per_patient(
        "beat_aml",
        candidates,
        records_by_sample,
        analysis_set="main",
    )


def file_sample_format(path: Path, headers: list[str]) -> str:
    sample_columns = matrix_sample_columns(headers)
    if sample_columns:
        return sample_id_format(sample_columns)
    sample_columns = {
        "sample_id",
        "tumor_sample_barcode",
        "patient_id",
    }
    preferred = next(
        (name for name in headers if name.casefold() in sample_columns), None
    )
    sample_values = [
        row[preferred] for row in iter_tsv(path) if preferred and row.get(preferred)
    ]
    return sample_id_format(sample_values)


def file_headers(path: Path) -> list[str]:
    return find_header(path)


def main(check_only: bool = False) -> int:
    rows: list[dict[str, str]] = []
    analysis_rows: list[dict[str, str]] = []
    cohorts = load_config()
    for cohort in cohorts:
        cohort_name = cohort["cohort"]
        study_dir = RAW_DIR / cohort["study_id"]
        for file_name in ("data_clinical_sample.txt", "data_clinical_patient.txt"):
            path = study_dir / file_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing required clinical file: {path.relative_to(ROOT)}"
                )
            headers = file_headers(path)
            print(
                f"DATA CHECK {DISPLAY_NAMES[cohort_name]} {file_name} columns: "
                f"{' | '.join(headers)}"
            )

    for cohort in cohorts:
        cohort_name = cohort["cohort"]
        study_id = cohort["study_id"]
        study_dir = RAW_DIR / study_id
        rna_file = cohort["rna_file"]
        file_names = (*FILES_COMMON[:2], rna_file, *FILES_COMMON[2:])
        if (study_dir / "data_sv.txt").is_file():
            file_names = (*file_names, "data_sv.txt")
        paths = {name: study_dir / name for name in file_names}
        missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing inputs for {study_id}: {', '.join(missing)}. Run `make data`."
            )

        clinical = {
            name: clinical_values(paths[name])
            for name in ("data_clinical_sample.txt", "data_clinical_patient.txt")
        }
        diagnosis, relapse = diagnosis_relapse_candidates(clinical)
        for name, path in paths.items():
            headers = file_headers(path)
            row = {
                "cohort": cohort["cohort"],
                "record_type": "file",
                "file": name,
                "header_columns": "|".join(headers),
                "sample_id_format": file_sample_format(path, headers),
                "diagnosis_column_candidates": candidate_text(diagnosis),
                "relapse_column_candidates": candidate_text(relapse),
            }
            rows.append(row)

        sample_headers = file_headers(paths["data_clinical_sample.txt"])
        patient_headers = file_headers(paths["data_clinical_patient.txt"])
        sample_id_col = find_column(sample_headers, "SAMPLE_ID")
        patient_id_col = find_column(patient_headers, "PATIENT_ID")
        sample_records = list(iter_tsv(paths["data_clinical_sample.txt"]))
        records_by_sample = {
            row[sample_id_col]: row for row in sample_records if row.get(sample_id_col)
        }
        sample_ids = set(records_by_sample)
        patient_ids = {
            row[patient_id_col]
            for row in iter_tsv(paths["data_clinical_patient.txt"])
            if row.get(patient_id_col)
        }
        rna_samples = set(matrix_sample_columns(file_headers(paths[rna_file])))
        mutation_panels = mutation_profile_panels(
            paths["data_gene_panel_matrix.txt"]
        )
        mutation_profiled_samples = {
            sample
            for sample, panel in mutation_panels.items()
            if panel.strip().upper() not in UNPROFILED_VALUES
        }
        eligible_samples = sample_ids & rna_samples & mutation_profiled_samples
        profiled, panel_names = mutation_profile_samples(
            paths["data_gene_panel_matrix.txt"], {"NPM1", "DNMT3A"}
        )
        if cohort_name == "beat_aml":
            print_data_check_values(
                cohort_name,
                sample_records,
                BEAT_AML_DISTINCT_FIELDS,
                rows,
                headers=sample_headers,
                file_name="data_clinical_sample.txt",
            )
            analysis_samples = select_beat_aml_samples(
                eligible_samples, records_by_sample
            )
        elif cohort_name == "tcga_laml":
            print_data_check_values(
                cohort_name,
                sample_records,
                ("ONCOTREE_CODE",),
                rows,
                headers=sample_headers,
                file_name="data_clinical_sample.txt",
            )
            patient_records = list(iter_tsv(paths["data_clinical_patient.txt"]))
            print_data_check_values(
                cohort_name,
                patient_records,
                TCGA_PATIENT_DISTINCT_FIELDS,
                rows,
                headers=patient_headers,
                file_name="data_clinical_patient.txt",
            )
            analysis_samples = eligible_samples
        else:
            raise ValueError(f"Unregistered cohort in config.tsv: {cohort_name}")

        if check_only:
            print(
                f"DATA CHECK {DISPLAY_NAMES[cohort_name]}: "
                f"{len(eligible_samples)} RNA+mutation-profiled candidates, "
                f"{len(analysis_samples)} analysis samples"
            )
            continue

        positives = positive_samples(
            paths["data_mutations.txt"], {"NPM1", "DNMT3A"}, profiled
        )
        analysis_patient_ids = {
            patient_id(cohort_name, sample) for sample in analysis_samples
        }

        rows.append(
            {
                "cohort": cohort_name,
                "record_type": "cohort_summary",
                "sample_count": str(len(sample_ids)),
                "patient_count": str(len(patient_ids)),
                "rna_sample_count": str(len(sample_ids & rna_samples)),
                "rna_and_mutation_profiled_sample_count": str(len(eligible_samples)),
                "notes": f"RNA file: {rna_file}",
            }
        )
        target_rna = rna_samples if cohort_name == "tcga_laml" else set()
        target_dnmt3a_profiled = target_rna & profiled["DNMT3A"]
        target_dnmt3a_positives = target_dnmt3a_profiled & positives["DNMT3A"]
        analysis_rows.append(
            {
                "cohort": cohort_name,
                "n_samples_with_rna_and_mutation_profiling": str(
                    len(eligible_samples)
                ),
                "n_samples_in_analysis_set": str(len(analysis_samples)),
                "n_patients": str(len(analysis_patient_ids)),
                "npm1_positive_count": str(
                    len(analysis_samples & positives["NPM1"])
                ),
                "dnmt3a_positive_count": str(
                    len(analysis_samples & positives["DNMT3A"])
                ),
                "dnmt3a_profiled_sample_count": str(
                    len(analysis_samples & profiled["DNMT3A"])
                ),
                "target_rna_sample_count": (
                    str(len(target_rna)) if cohort_name == "tcga_laml" else ""
                ),
                "target_rna_dnmt3a_profiled_sample_count": (
                    str(len(target_dnmt3a_profiled))
                    if cohort_name == "tcga_laml"
                    else ""
                ),
                "target_rna_dnmt3a_positive_count": (
                    str(len(target_dnmt3a_positives))
                    if cohort_name == "tcga_laml"
                    else ""
                ),
                "dnmt3a_secondary_eligible": (
                    str(len(target_dnmt3a_positives) >= 25).lower()
                    if cohort_name == "tcga_laml"
                    else ""
                ),
                "selected_sample_ids": "|".join(sorted(analysis_samples)),
            }
        )
        for gene in ("NPM1", "DNMT3A"):
            rows.append(
                {
                    "cohort": cohort_name,
                    "record_type": "gene_panel",
                    "gene": gene,
                    "panel_covered": str(bool(profiled[gene])).lower(),
                    "profiled_sample_count": str(len(profiled[gene])),
                    "positive_sample_count": str(len(positives[gene])),
                    "notes": (
                        f"panels={','.join(sorted(panel_names[gene])) or 'unknown'}; "
                        "positives restricted to covered samples"
                    ),
                }
            )

    if check_only:
        return 0

    fields = (
        "cohort",
        "record_type",
        "file",
        "header_columns",
        "sample_id_format",
        "diagnosis_column_candidates",
        "relapse_column_candidates",
        "sample_count",
        "patient_count",
        "rna_sample_count",
        "rna_and_mutation_profiled_sample_count",
        "gene",
        "panel_covered",
        "profiled_sample_count",
        "positive_sample_count",
        "notes",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    analysis_fields = (
        "cohort",
        "n_samples_with_rna_and_mutation_profiling",
        "n_samples_in_analysis_set",
        "n_patients",
        "npm1_positive_count",
        "dnmt3a_positive_count",
        "dnmt3a_profiled_sample_count",
        "target_rna_sample_count",
        "target_rna_dnmt3a_profiled_sample_count",
        "target_rna_dnmt3a_positive_count",
        "dnmt3a_secondary_eligible",
        "selected_sample_ids",
    )
    with ANALYSIS_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=analysis_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(analysis_rows)
    output_name = OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT
    print(f"Wrote {output_name} ({len(rows)} rows)")
    analysis_output_name = (
        ANALYSIS_OUT.relative_to(ROOT)
        if ANALYSIS_OUT.is_relative_to(ROOT)
        else ANALYSIS_OUT
    )
    print(f"Wrote {analysis_output_name} ({len(analysis_rows)} cohort rows)")
    for row in rows:
        if row.get("record_type") == "cohort_summary":
            print(
                f"{row['cohort']}: samples={row['sample_count']}, "
                f"patients={row['patient_count']}, RNA-seq samples={row['rna_sample_count']}"
            )
        elif row.get("record_type") == "gene_panel":
            print(
                f"{row['cohort']} {row['gene']}: covered={row['panel_covered']}, "
                f"profiled={row['profiled_sample_count']}, "
                f"positive={row['positive_sample_count']}"
            )
    for row in analysis_rows:
        print(
            f"ANALYSIS SET {row['cohort']}: RNA+mutation-profiled="
            f"{row['n_samples_with_rna_and_mutation_profiling']}, "
            f"analysis samples={row['n_samples_in_analysis_set']}, "
            f"patients={row['n_patients']}, NPM1+={row['npm1_positive_count']}, "
            f"DNMT3A+={row['dnmt3a_positive_count']}"
        )
        if row["cohort"] == "tcga_laml":
            print(
                "DATA CHECK TCGA-LAML DNMT3A gate: positives among RNA samples="
                f"{row['target_rna_dnmt3a_positive_count']} "
                f"(>=25: {row['dnmt3a_secondary_eligible']})"
            )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="print schema and sample-selection checks without mutation counts",
    )
    raise SystemExit(main(parser.parse_args().check_only))
