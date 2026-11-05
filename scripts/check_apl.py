#!/usr/bin/env python3
"""Read-only APL evidence counts for full cohorts and main analysis sets."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohortshift.data import (  # noqa: E402
    apl_evidence_samples,
    iter_tsv,
    load_expression_matrix,
    mutation_profile_panels,
    select_one_sample_per_patient,
)

CONFIG = ROOT / "config.tsv"
RAW_DIR = ROOT / "data" / "raw"
UNPROFILED_VALUES = {"", "NA", "N/A", "NULL", "NONE"}
EVIDENCE_NAMES = (
    ("clinical_CANCER_TYPE_DETAILED", "clinical_cancer_type_detailed"),
    ("SV_partner_genes", "sv_partner_genes"),
    ("SV_Event_Info", "sv_event_info"),
    ("union", "union"),
)


def _read_config() -> list[dict[str, str]]:
    with CONFIG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _clinical_by_sample(path: Path) -> dict[str, dict[str, str]]:
    rows = list(iter_tsv(path))
    if not rows:
        raise ValueError(f"No clinical sample records found in {path}")
    sample_column = next(
        (column for column in rows[0] if column.upper() == "SAMPLE_ID"), None
    )
    if sample_column is None:
        raise ValueError(f"{path} has no SAMPLE_ID column")
    return {
        row[sample_column]: row for row in rows if row.get(sample_column)
    }


def _print_counts(
    cohort: str, scope: str, samples: set[str], evidence: dict[str, set[str]]
) -> None:
    counts = [
        f"{label}={len(evidence[key] & samples)}"
        for label, key in EVIDENCE_NAMES
    ]
    print(f"APL CHECK {cohort} {scope}: " + "; ".join(counts))


def main() -> int:
    for config in _read_config():
        cohort = config["cohort"]
        study_dir = RAW_DIR / config["study_id"]
        expression_path = study_dir / config["rna_file"]
        clinical_path = study_dir / "data_clinical_sample.txt"
        panel_path = study_dir / "data_gene_panel_matrix.txt"
        structural_variant_path = study_dir / "data_sv.txt"
        required = (
            expression_path,
            clinical_path,
            panel_path,
            structural_variant_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing required APL check inputs: {', '.join(missing)}"
            )
        if structural_variant_path.stat().st_size == 0:
            raise ValueError(
                f"Structural-variant file is empty: {structural_variant_path}"
            )

        clinical_by_sample = _clinical_by_sample(clinical_path)
        structural_variant_rows = list(iter_tsv(structural_variant_path))
        if not structural_variant_rows:
            raise ValueError(
                f"Structural-variant file has no records: {structural_variant_path}"
            )
        columns, evidence = apl_evidence_samples(
            clinical_by_sample,
            structural_variant_rows,
            cohort=cohort,
        )
        print(f"APL CHECK {cohort} data_sv.txt columns: {' | '.join(columns)}")

        expression_samples = set(load_expression_matrix(expression_path).index)
        panel_values = mutation_profile_panels(panel_path)
        mutation_profiled = {
            sample
            for sample, panel in panel_values.items()
            if panel.strip().upper() not in UNPROFILED_VALUES
        }
        candidates = expression_samples & set(clinical_by_sample) & mutation_profiled
        selected = select_one_sample_per_patient(
            cohort, candidates, clinical_by_sample, analysis_set="main"
        )
        _print_counts(cohort, "full_file", evidence["union"], evidence)
        _print_counts(cohort, "main_analysis_set", selected, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
