"""Utilities for parsing cBioPortal tab-separated files."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pandas as pd

_HEADER_TOKENS = {
    "PATIENT_ID",
    "SAMPLE_ID",
    "HUGO_SYMBOL",
    "TUMOR_SAMPLE_BARCODE",
}
_ANNOTATION_COLUMNS = {
    "HUGO_SYMBOL",
    "ENTREZ_GENE_ID",
    "GENE",
    "GENE_ID",
    "GENE_NAME",
    "CHROMOSOME",
    "START_POSITION",
    "END_POSITION",
}
_BROAD_SEQUENCING_PANELS = {"WXS", "WGS", "WHOLE_EXOME", "WHOLE_GENOME"}
_UNPROFILED_PANEL_VALUES = {"", "NA", "N/A", "NULL", "NONE"}
_PATIENT_ID_SUFFIXES = {
    "beat_aml": re.compile(r"_BA\d{4}$"),
    "tcga_laml": re.compile(r"-03$"),
}


def _clean_header(fields: list[str]) -> list[str]:
    if fields:
        fields[0] = fields[0].lstrip("#").strip()
    return [field.strip().lstrip("#") for field in fields]


def find_header(path: str | Path) -> list[str]:
    """Find a cBioPortal header, allowing metadata comments before it."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = _clean_header(next(csv.reader([line], delimiter="\t")))
            normalized = {field.upper() for field in fields}
            if normalized & _HEADER_TOKENS:
                return fields
    raise ValueError(f"No recognizable tabular header found in {path}")


def iter_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield records after the first recognizable cBioPortal header."""
    path = Path(path)
    header = find_header(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            fields = _clean_header(next(csv.reader([line], delimiter="\t")))
            if fields == header:
                break
        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")
        for row in reader:
            if not row or all(value is None or value == "" for value in row.values()):
                continue
            yield {key: (value or "").strip() for key, value in row.items() if key}


def matrix_sample_columns(headers: list[str]) -> list[str]:
    """Return expression/panel sample columns after gene annotation columns."""
    if "SAMPLE_ID" in {name.upper() for name in headers}:
        return []
    return [
        name
        for name in headers
        if name.upper() not in _ANNOTATION_COLUMNS
        and name.upper() not in {"PATIENT_ID", "SAMPLE_ID"}
    ]


def mutation_profile_panels(path: str | Path) -> dict[str, str]:
    """Return each sample's mutations assay panel from the cBioPortal matrix."""
    headers = find_header(path)
    sample_col = next(
        (name for name in headers if name.upper() == "SAMPLE_ID"), None
    )
    mutation_col = next(
        (name for name in headers if name.casefold() == "mutations"), None
    )
    if sample_col is None or mutation_col is None:
        raise ValueError(
            f"{path} must have SAMPLE_ID and mutations columns for gene-panel profiling"
        )
    return {
        row[sample_col]: row[mutation_col].strip()
        for row in iter_tsv(path)
        if row.get(sample_col)
    }


def mutation_profile_samples(
    path: str | Path, genes: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Map known broad-sequencing mutation profiles to target-gene coverage.

    The cBioPortal gene-panel matrix is sample-oriented. A WXS/WGS panel is
    treated as covering the requested exonic genes; unknown targeted-panel IDs
    are not assumed to include them.
    """
    profiled = {gene: set() for gene in genes}
    panel_names: dict[str, set[str]] = {gene: set() for gene in genes}
    for sample, panel in mutation_profile_panels(path).items():
        if not sample or panel.upper() not in _BROAD_SEQUENCING_PANELS:
            continue
        for gene in genes:
            profiled[gene].add(sample)
            panel_names[gene].add(panel)
    return profiled, panel_names


def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    """Load cBioPortal gene-by-sample expression as sample-by-gene values.

    Duplicate normalized Hugo symbols are resolved by keeping the row with
    highest mean expression across samples.
    """
    path = Path(path)
    raw = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    symbol_col = next(
        (column for column in raw.columns if str(column).casefold() == "hugo_symbol"),
        None,
    )
    if symbol_col is None:
        raise ValueError(f"{path} has no Hugo_Symbol column")
    sample_columns = matrix_sample_columns([str(column) for column in raw.columns])
    if not sample_columns:
        raise ValueError(f"{path} has no expression sample columns")

    values = raw[sample_columns].apply(pd.to_numeric, errors="coerce")
    symbols = raw[symbol_col].astype("string").str.strip().str.upper()
    valid_symbols = (
        symbols.notna()
        & symbols.ne("")
        & symbols.ne("NAN")
        & symbols.ne("NA")
    )
    values = values.loc[valid_symbols].copy()
    values.index = symbols.loc[valid_symbols]
    means = values.mean(axis=1, skipna=True)
    keep_indices = (
        pd.DataFrame(
            {
                "symbol": values.index,
                "mean": means.to_numpy(),
                "position": range(len(values)),
            }
        )
        .sort_values(["symbol", "mean", "position"], ascending=[True, False, True])
        .drop_duplicates("symbol", keep="first")["position"]
        .to_numpy()
    )
    deduplicated = values.iloc[keep_indices].copy()
    deduplicated.index.name = "Hugo_Symbol"
    if deduplicated.index.has_duplicates:
        raise RuntimeError(f"Duplicate Hugo symbols remain in {path}")
    return deduplicated.T.astype(float)


def parse_patient_id(cohort: str, sample_id: str) -> str:
    """Remove the preregistered trailing sample suffix for a cohort."""
    try:
        suffix = _PATIENT_ID_SUFFIXES[cohort]
    except KeyError as error:
        raise ValueError(f"Unknown cohort for patient ID parsing: {cohort}") from error
    patient, count = suffix.subn("", sample_id)
    if count != 1:
        raise ValueError(
            f"Sample ID {sample_id!r} does not match the registered "
            f"{cohort} patient-ID suffix"
        )
    return patient


def detect_apl_samples(
    clinical_by_sample: dict[str, dict[str, str]],
    structural_variant_rows: Iterable[dict[str, str]] | None,
    *,
    cohort: str,
) -> set[str]:
    """Find APL samples from clinical diagnosis or PML-RARA structural variants."""
    display_cohort = {
        "beat_aml": "Beat AML",
        "tcga_laml": "TCGA-LAML",
    }.get(cohort)
    if display_cohort is None:
        raise ValueError(f"Unknown cohort for APL detection: {cohort}")
    if structural_variant_rows is None:
        raise ValueError(f"{display_cohort} e6a requires data_sv.txt")

    diagnosis_column = "CANCER_TYPE_DETAILED"
    if clinical_by_sample and any(
        diagnosis_column not in row for row in clinical_by_sample.values()
    ):
        raise ValueError(
            f"Clinical sample records must include {diagnosis_column} to classify APL"
        )

    clinical_apl_samples = {
        sample
        for sample, row in clinical_by_sample.items()
        if _is_apl_diagnosis(row.get(diagnosis_column, ""))
    }
    apl_samples = set(clinical_apl_samples)

    rows = list(structural_variant_rows)
    if not rows:
        raise ValueError(f"{display_cohort} e6a requires non-empty data_sv.txt")
    columns = {column.casefold(): column for column in rows[0]}
    sample_column = columns.get("sample_id")
    site1_column = columns.get("site1_hugo_symbol")
    site2_column = columns.get("site2_hugo_symbol")
    event_column = columns.get("event_info")
    if sample_column is None or (
        (site1_column is None or site2_column is None) and event_column is None
    ):
        raise ValueError(
            "Structural-variant records must include Sample_Id and either "
            "Site1_Hugo_Symbol/Site2_Hugo_Symbol or Event_Info"
        )
    print(
        f"DATA CHECK {display_cohort} data_sv.txt columns: "
        f"{' | '.join(rows[0])}"
    )

    partner_samples: set[str] = set()
    event_samples: set[str] = set()
    for row in rows:
        sample = row.get(sample_column, "").strip()
        partners = {
            row.get(column, "").strip().upper()
            for column in (site1_column, site2_column)
            if column is not None
        }
        event = row.get(event_column, "").casefold() if event_column else ""
        if sample and partners == {"PML", "RARA"}:
            partner_samples.add(sample)
        if sample and "pml-rara" in event:
            event_samples.add(sample)
    apl_samples.update(partner_samples | event_samples)
    print(
        f"DATA CHECK {display_cohort} PML-RARA sample counts: "
        f"clinical_diagnosis={len(clinical_apl_samples)}; "
        f"sv_partner_genes={len(partner_samples)}; "
        f"sv_event_info={len(event_samples)}; "
        f"combined={len(apl_samples)}"
    )
    return apl_samples


def _is_apl_diagnosis(value: str) -> bool:
    normalized = value.casefold()
    return (
        "acute promyelocytic" in normalized
        or "pml-rara" in normalized
        or re.search(r"\bapl\b", normalized) is not None
    )


def select_one_sample_per_patient(
    cohort: str,
    candidate_samples: set[str],
    clinical_by_sample: dict[str, dict[str, str]],
    *,
    analysis_set: Literal["main", "e6a", "e6b"] = "main",
    apl_samples: set[str] | None = None,
) -> set[str]:
    """Apply the registered cohort-specific one-sample-per-patient rule."""
    if analysis_set not in {"main", "e6a", "e6b"}:
        raise ValueError(f"Unknown analysis set: {analysis_set}")
    if cohort not in {"beat_aml", "tcga_laml"}:
        raise ValueError(f"Unknown cohort for sample selection: {cohort}")
    if analysis_set == "e6a" and apl_samples is None:
        raise ValueError("e6a selection requires APL sample IDs from clinical/SV data")
    required_columns = set()
    if cohort == "beat_aml":
        required_columns.update(
            {
                "DISEASE_STAGE_AT_SPECIMEN_COLLECTION",
                "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION",
            }
        )
    if analysis_set == "e6a":
        required_columns.add("ONCOTREE_CODE")
    missing_columns = sorted(
        column
        for column in required_columns
        if clinical_by_sample
        and any(column not in row for row in clinical_by_sample.values())
    )
    if missing_columns:
        raise ValueError(
            f"{cohort} sample selection requires clinical columns: "
            + ", ".join(missing_columns)
        )

    candidates = set(candidate_samples)
    if analysis_set == "e6a":
        candidates = {
            sample
            for sample in candidates
            if clinical_by_sample[sample].get("ONCOTREE_CODE", "").strip() == "AML"
            and sample not in apl_samples
        }
    if cohort == "beat_aml" and analysis_set in {"main", "e6a"}:
        candidates = {
            sample
            for sample in candidates
            if clinical_by_sample[sample].get(
                "DISEASE_STAGE_AT_SPECIMEN_COLLECTION"
            )
            == "Initial Diagnosis"
        }

    by_patient: dict[str, list[str]] = {}
    for sample in candidates:
        by_patient.setdefault(parse_patient_id(cohort, sample), []).append(sample)

    selected: set[str] = set()
    for samples in by_patient.values():
        if cohort == "beat_aml":
            selected.add(
                min(
                    samples,
                    key=lambda sample: (
                        0
                        if analysis_set == "e6b"
                        and clinical_by_sample[sample].get(
                            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION"
                        )
                        == "Initial Diagnosis"
                        else 1,
                        _collection_time_key(
                            clinical_by_sample[sample].get(
                                "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION", ""
                            )
                        ),
                        sample,
                    ),
                )
            )
        else:
            selected.add(min(samples))
    return selected


def _collection_time_key(value: str) -> tuple[int, float]:
    cleaned = value.strip()
    if cleaned.upper() in _UNPROFILED_PANEL_VALUES:
        return (1, float("inf"))
    try:
        numeric_value = float(cleaned)
    except ValueError as error:
        raise ValueError(
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION must be numeric "
            f"or missing; found {value!r}"
        ) from error
    if not np.isfinite(numeric_value):
        raise ValueError(
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION must be finite; "
            f"found {value!r}"
        )
    return (0, numeric_value)


def split_patient_ids(
    patient_ids: set[str], *, seed: int
) -> dict[str, str]:
    """Assign patients to deterministic 60/20/20 train/validation/test splits."""
    ordered = sorted(patient_ids)
    if not ordered:
        raise ValueError("cannot split an empty patient set")
    shuffled = np.random.default_rng(seed).permutation(ordered).tolist()
    count = len(shuffled)
    if count == 1:
        counts = (1, 0, 0)
    elif count == 2:
        counts = (1, 0, 1)
    else:
        validation_count = max(1, int(np.floor(0.20 * count)))
        test_count = max(1, int(np.floor(0.20 * count)))
        counts = (count - validation_count - test_count, validation_count, test_count)
    train_end = counts[0]
    validation_end = train_end + counts[1]
    return {
        patient: (
            "train"
            if position < train_end
            else "validation"
            if position < validation_end
            else "test"
        )
        for position, patient in enumerate(shuffled)
    }


def split_patient_ids_stratified(
    labels_by_patient: dict[str, int | None], *, seed: int
) -> dict[str, str]:
    """Split patients 60/20/20 within each observed binary-label stratum."""
    if not labels_by_patient:
        raise ValueError("cannot split an empty patient-label mapping")
    strata: dict[int | None, list[str]] = {0: [], 1: [], None: []}
    for patient, label in labels_by_patient.items():
        if label not in strata:
            raise ValueError(f"Expected binary or missing label, got {label!r}")
        strata[label].append(patient)

    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for label in (0, 1, None):
        patients = strata[label]
        if not patients:
            continue
        shuffled = rng.permutation(sorted(patients)).tolist()
        count = len(shuffled)
        if count == 1:
            split_counts = (1, 0, 0)
        elif count == 2:
            split_counts = (1, 0, 1)
        else:
            validation_count = max(1, int(np.floor(0.20 * count)))
            test_count = max(1, int(np.floor(0.20 * count)))
            split_counts = (
                count - validation_count - test_count,
                validation_count,
                test_count,
            )
        train_end = split_counts[0]
        validation_end = train_end + split_counts[1]
        for position, patient in enumerate(shuffled):
            assignments[patient] = (
                "train"
                if position < train_end
                else "validation"
                if position < validation_end
                else "test"
            )
    return assignments


def sample_id_format(values: list[str]) -> str:
    """Describe common identifier shape and provide a concrete example."""
    valid = [value for value in values if value]
    if not valid:
        return "not present / no sample identifiers detected"
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for value in valid:
        shape = "".join(
            "#" if character.isdigit() else "X" if character.isalpha() else character
            for character in value
        )
        counts[shape] = counts.get(shape, 0) + 1
        examples.setdefault(shape, value)
    common = max(counts, key=counts.get)
    return f"example={examples[common]}; pattern={common}; n={len(set(valid))}"
