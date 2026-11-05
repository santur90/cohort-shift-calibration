#!/usr/bin/env python3
"""Audit Beat AML selection and preregistered mutation counts; writes one TSV."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.data import (  # noqa: E402
    iter_tsv,
    load_expression_matrix,
    mutation_profile_samples,
    parse_patient_id,
    select_one_sample_per_patient,
)
from cohortshift.labels import labels_from_maf_rows  # noqa: E402

BEAT_AML_DIR = ROOT / "data" / "raw" / "aml_ohsu_2022"
OUTPUT = ROOT / "results" / "tables" / "selection_check.tsv"
GENES = ("NPM1", "DNMT3A")
ONCOTREE_CODES = ("AML", "ALAL", "MBN", "MDS", "MDS/MPN", "MPN")
STAGE_COLUMN = "DISEASE_STAGE_AT_SPECIMEN_COLLECTION"
SAMPLE_COLUMN = "SAMPLE_ID"
PATIENT_COLUMN = "PATIENT_ID"


def _stage(row: dict[str, str]) -> str:
    value = row.get(STAGE_COLUMN, "").strip()
    return value if value and value.upper() != "NA" else "NA"


def _write_row(
    writer: csv.DictWriter,
    *,
    section: str,
    subset: str,
    n: int,
    positives: int | str = "",
    prevalence: float | str = "",
    stage: str = "",
    oncotree_code: str = "",
    gene: str = "",
    patients: int | str = "",
) -> None:
    writer.writerow(
        {
            "section": section,
            "subset": subset,
            "stage": stage,
            "oncotree_code": oncotree_code,
            "gene": gene,
            "n": n,
            "patients": patients,
            "positives": positives,
            "prevalence": prevalence,
        }
    )


def _metric_rows(
    writer: csv.DictWriter,
    labels,
    subsets: dict[str, set[str]],
) -> None:
    for subset_name, samples in subsets.items():
        for gene in GENES:
            label_values = labels.loc[sorted(samples), gene]
            if label_values.isna().any():
                missing = int(label_values.isna().sum())
                raise ValueError(
                    f"{subset_name} contains {missing} samples without {gene} "
                    "mutation profiling"
                )
            n = len(label_values)
            positives = int(label_values.eq(1).sum())
            prevalence = positives / n if n else 0.0
            _write_row(
                writer,
                section="mutation_counts",
                subset=subset_name,
                n=n,
                positives=positives,
                prevalence=prevalence,
                gene=gene,
            )
            print(
                f"{subset_name} {gene}: N={n}, positives={positives}, "
                f"prevalence={prevalence:.6f}"
            )


def main() -> int:
    expression = load_expression_matrix(
        BEAT_AML_DIR / "data_mrna_seq_rpkm.txt"
    )
    clinical = list(iter_tsv(BEAT_AML_DIR / "data_clinical_sample.txt"))
    records_by_sample = {
        row[SAMPLE_COLUMN]: row for row in clinical if row.get(SAMPLE_COLUMN)
    }
    profiled, _ = mutation_profile_samples(
        BEAT_AML_DIR / "data_gene_panel_matrix.txt", set(GENES)
    )
    candidates = set(expression.index) & set(records_by_sample)
    candidates &= set.intersection(*(profiled[gene] for gene in GENES))
    selected = select_one_sample_per_patient(
        "beat_aml",
        candidates,
        records_by_sample,
        analysis_set="e6b",
    )
    patient_by_sample = {
        sample: parse_patient_id("beat_aml", sample) for sample in candidates
    }

    stages_by_candidate_patient: dict[str, set[str]] = defaultdict(set)
    candidates_by_stage: dict[str, set[str]] = defaultdict(set)
    candidate_patients_by_stage: dict[str, set[str]] = defaultdict(set)
    for sample in candidates:
        stage = _stage(records_by_sample[sample])
        patient = patient_by_sample[sample]
        candidates_by_stage[stage].add(sample)
        candidate_patients_by_stage[stage].add(patient)
        stages_by_candidate_patient[patient].add(stage)

    selected_by_stage: dict[str, set[str]] = defaultdict(set)
    selected_by_code: dict[str, set[str]] = defaultdict(set)
    selected_cross: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in selected:
        row = records_by_sample[sample]
        stage = _stage(row)
        code = row.get("ONCOTREE_CODE", "").strip() or "NA"
        selected_by_stage[stage].add(sample)
        selected_by_code[code].add(sample)
        selected_cross[(code, stage)].add(sample)

    fallback_patients = {
        patient
        for patient, stages in stages_by_candidate_patient.items()
        if "Initial Diagnosis" not in stages
    }
    mutation_rows = list(iter_tsv(BEAT_AML_DIR / "data_mutations.txt"))
    labels = labels_from_maf_rows(
        mutation_rows,
        set(GENES),
        selected,
        profiled,
    )
    initial_diagnosis = selected_by_stage.get("Initial Diagnosis", set())
    diagnosis_aml = initial_diagnosis & selected_by_code.get("AML", set())
    metric_subsets = {
        "all_selected": selected,
        "initial_diagnosis": initial_diagnosis,
        "initial_diagnosis_AML": diagnosis_aml,
    }

    fields = (
        "section",
        "subset",
        "stage",
        "oncotree_code",
        "gene",
        "n",
        "patients",
        "positives",
        "prevalence",
    )
    if not OUTPUT.parent.is_dir():
        raise FileNotFoundError(
            f"Expected output directory does not exist: {OUTPUT.parent}"
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for stage in sorted(candidates_by_stage):
            n = len(candidates_by_stage[stage])
            patients = len(candidate_patients_by_stage[stage])
            _write_row(
                writer,
                section="candidate_stage_counts",
                subset="RNA_and_mutation_profiled",
                stage=stage,
                n=n,
                patients=patients,
            )
            print(
                f"candidates stage={stage}: samples={n}, patients={patients}"
            )

        print(
            "fallback patients with no Initial Diagnosis candidate: "
            f"{len(fallback_patients)}"
        )
        _write_row(
            writer,
            section="fallback_patients",
            subset="no_Initial_Diagnosis_candidate",
            n=len(fallback_patients),
            patients=len(fallback_patients),
        )

        for stage in sorted(selected_by_stage):
            n = len(selected_by_stage[stage])
            patients = len(
                {patient_by_sample[sample] for sample in selected_by_stage[stage]}
            )
            _write_row(
                writer,
                section="selected_stage_counts",
                subset="selected_analysis_set",
                stage=stage,
                n=n,
                patients=patients,
            )
            print(f"selected stage={stage}: samples={n}")

        for code in ONCOTREE_CODES:
            samples = selected_by_code.get(code, set())
            _write_row(
                writer,
                section="selected_oncotree_counts",
                subset="selected_analysis_set",
                oncotree_code=code,
                n=len(samples),
                patients=len({patient_by_sample[sample] for sample in samples}),
            )
            print(f"selected ONCOTREE_CODE={code}: samples={len(samples)}")

        for code in ONCOTREE_CODES:
            for stage in sorted(selected_by_stage):
                samples = selected_cross.get((code, stage), set())
                _write_row(
                    writer,
                    section="oncotree_by_selected_stage",
                    subset="selected_analysis_set",
                    oncotree_code=code,
                    stage=stage,
                    n=len(samples),
                )
                print(f"cross-table {code} x {stage}: {len(samples)}")

        _metric_rows(writer, labels, metric_subsets)

    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
