"""Mutation-label and clinical timepoint candidate helpers."""

from __future__ import annotations

import re

_DIAGNOSIS_TERMS = re.compile(
    r"\b(diagnos\w*|initial|baseline|presentation|de novo|primary)\b", re.I
)
_RELAPSE_TERMS = re.compile(
    r"\b(relaps\w*|recur\w*|progress\w*|refractory)\b", re.I
)
NON_SILENT_VARIANT_CLASSIFICATIONS = frozenset(
    {
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Splice_Site",
        "Translation_Start_Site",
        "Nonstop_Mutation",
    }
)


def is_non_silent_variant(classification: str) -> bool:
    """Return whether a MAF variant classification is in the registered set."""
    return classification in NON_SILENT_VARIANT_CLASSIFICATIONS


def diagnosis_relapse_candidates(
    clinical_tables: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """List clinical fields whose names or values suggest diagnosis/relapse."""
    diagnosis: dict[str, list[str]] = {}
    relapse: dict[str, list[str]] = {}
    for table, columns in clinical_tables.items():
        for column, values in columns.items():
            name = column.lstrip("#")
            diagnosis_matches = sorted(
                {value for value in values if _DIAGNOSIS_TERMS.search(value)}
            )
            relapse_matches = sorted(
                {value for value in values if _RELAPSE_TERMS.search(value)}
            )
            if _DIAGNOSIS_TERMS.search(name) or diagnosis_matches:
                diagnosis[f"{table}:{name}"] = diagnosis_matches[:8]
            if _RELAPSE_TERMS.search(name) or relapse_matches:
                relapse[f"{table}:{name}"] = relapse_matches[:8]
    return diagnosis, relapse


def mutation_labels(
    mutations: set[str], profiled_samples: set[str], all_samples: set[str]
) -> dict[str, int | None]:
    """Encode positives as one, profiled negatives as zero, unprofiled as None."""
    return {
        sample: int(sample in mutations) if sample in profiled_samples else None
        for sample in all_samples
    }
