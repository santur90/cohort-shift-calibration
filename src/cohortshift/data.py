"""Utilities for parsing cBioPortal tab-separated files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

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


def mutation_profile_samples(
    path: str | Path, genes: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Map known broad-sequencing mutation profiles to target-gene coverage.

    The cBioPortal gene-panel matrix is sample-oriented. A WXS/WGS panel is
    treated as covering the requested exonic genes; unknown targeted-panel IDs
    are not assumed to include them.
    """
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
    profiled = {gene: set() for gene in genes}
    panel_names: dict[str, set[str]] = {gene: set() for gene in genes}
    for row in iter_tsv(path):
        sample = row.get(sample_col, "")
        panel = row.get(mutation_col, "").strip()
        if not sample or panel.upper() not in _BROAD_SEQUENCING_PANELS:
            continue
        for gene in genes:
            profiled[gene].add(sample)
            panel_names[gene].add(panel)
    return profiled, panel_names


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
