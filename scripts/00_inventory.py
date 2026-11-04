#!/usr/bin/env python3
"""Summarize locally fetched cBioPortal AML inputs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.data import (  # noqa: E402
    find_header,
    iter_tsv,
    matrix_sample_columns,
    mutation_profile_samples,
    sample_id_format,
)
from cohortshift.labels import diagnosis_relapse_candidates  # noqa: E402

CONFIG = ROOT / "config.tsv"
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables" / "inventory.tsv"

FILES_COMMON = (
    "data_clinical_sample.txt",
    "data_clinical_patient.txt",
    "data_mutations.txt",
    "data_gene_panel_matrix.txt",
)
NON_SILENT_EXCLUSIONS = {
    "3'flank",
    "3'utr",
    "5'flank",
    "5'utr",
    "downstream_gene_variant",
    "intron",
    "intergenic",
    "igr",
    "lincRNA",
    "non_coding_transcript_exon_variant",
    "rna",
    "silent",
    "splice_region",
    "synonymous",
    "upstream_gene_variant",
}
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


def non_silent(classification: str) -> bool:
    value = classification.strip()
    if not value:
        return False
    normalized = value.replace(" ", "_").casefold()
    return normalized not in {item.casefold() for item in NON_SILENT_EXCLUSIONS}


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
            and non_silent(row.get(variant_col, ""))
        ):
            positives[gene].add(sample)
    return positives


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


def main() -> int:
    rows: list[dict[str, str]] = []
    for cohort in load_config():
        study_id = cohort["study_id"]
        study_dir = RAW_DIR / study_id
        rna_file = cohort["rna_file"]
        file_names = (*FILES_COMMON[:2], rna_file, *FILES_COMMON[2:])
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
        sample_ids = {
            row[sample_id_col]
            for row in iter_tsv(paths["data_clinical_sample.txt"])
            if row.get(sample_id_col)
        }
        patient_ids = {
            row[patient_id_col]
            for row in iter_tsv(paths["data_clinical_patient.txt"])
            if row.get(patient_id_col)
        }
        rna_samples = set(matrix_sample_columns(file_headers(paths[rna_file])))
        profiled, panel_names = mutation_profile_samples(
            paths["data_gene_panel_matrix.txt"], {"NPM1", "DNMT3A"}
        )
        covered = {gene for gene, samples in profiled.items() if samples}
        positives = positive_samples(
            paths["data_mutations.txt"], {"NPM1", "DNMT3A"}, profiled
        )

        rows.append(
            {
                "cohort": cohort["cohort"],
                "record_type": "cohort_summary",
                "sample_count": str(len(sample_ids)),
                "patient_count": str(len(patient_ids)),
                "rna_sample_count": str(len(sample_ids & rna_samples)),
                "notes": f"RNA file: {rna_file}",
            }
        )
        for gene in ("NPM1", "DNMT3A"):
            rows.append(
                {
                    "cohort": cohort["cohort"],
                    "record_type": "gene_panel",
                    "gene": gene,
                    "panel_covered": str(gene in covered).lower(),
                    "profiled_sample_count": str(len(profiled[gene])),
                    "positive_sample_count": str(len(positives[gene])),
                    "notes": (
                        f"panels={','.join(sorted(panel_names[gene])) or 'unknown'}; "
                        "positives restricted to covered samples"
                    ),
                }
            )

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
    output_name = OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT
    print(f"Wrote {output_name} ({len(rows)} rows)")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
