import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("analysis_set", ["e6a", "e6b"])
def test_sensitivity_preparation_never_overwrites_main_outputs(
    tmp_path, monkeypatch, analysis_set
):
    script = Path(__file__).resolve().parents[1] / "scripts" / "01_prepare.py"
    spec = importlib.util.spec_from_file_location("prepare_outputs", script)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    prepare.ROOT = tmp_path
    prepare.RAW_DIR = tmp_path / "raw"
    prepare.PROCESSED_DIR = tmp_path / "data" / "processed"
    prepare.SPLITS_DIR = tmp_path / "results" / "splits"
    prepare.RESULTS_TABLES = tmp_path / "results" / "tables"
    cohorts = ("beat_aml", "tcga_laml")
    configs = [
        {"cohort": cohort, "study_id": cohort, "rna_file": "rna.txt"}
        for cohort in cohorts
    ]
    for config in configs:
        study = prepare.RAW_DIR / config["study_id"]
        study.mkdir(parents=True)
        for name in (
            "rna.txt", "data_clinical_sample.txt", "data_clinical_patient.txt",
            "data_mutations.txt", "data_gene_panel_matrix.txt", "data_sv.txt",
        ):
            (study / name).touch()
    genes = [f"G{i}" for i in range(2000)]
    samples = {
        cohort: [f"{cohort}_{i}" for i in range(12)]
        for cohort in cohorts
    }

    def clinical(path):
        cohort = path.parent.name
        if path.name == "data_clinical_sample.txt":
            return iter([
                {"SAMPLE_ID": sample, "ONCOTREE_CODE": "AML"}
                for sample in samples[cohort]
            ])
        return iter([{"SUBTYPE": "AML"}])

    monkeypatch.setattr(prepare, "_read_config", lambda: configs)
    monkeypatch.setattr(prepare, "iter_tsv", clinical)
    monkeypatch.setattr(prepare, "_print_distinct", lambda *args: None)
    monkeypatch.setattr(prepare, "_print_beat_cross_check", lambda *args: None)
    monkeypatch.setattr(
        prepare, "load_expression_matrix",
        lambda path: pd.DataFrame(
            np.ones((12, 2000)), index=samples[path.parent.name], columns=genes
        ),
    )
    monkeypatch.setattr(
        prepare, "mutation_profile_panels",
        lambda path: dict.fromkeys(samples[path.parent.name], "WXS"),
    )
    monkeypatch.setattr(prepare, "mutation_profile_samples", lambda *args: ({}, {}))
    monkeypatch.setattr(prepare, "detect_apl_samples", lambda *args, **kwargs: set())
    monkeypatch.setattr(
        prepare, "select_one_sample_per_patient",
        lambda cohort, candidates, clinical, **kwargs: candidates,
    )
    monkeypatch.setattr(prepare, "parse_patient_id", lambda cohort, sample: sample)
    monkeypatch.setattr(
        prepare, "labels_from_maf_rows",
        lambda rows, genes, selected, profiled: pd.DataFrame(
            {"NPM1": [i % 2 for i in range(len(selected))],
             "DNMT3A": [i % 2 for i in range(len(selected))]},
            index=sorted(selected),
        ),
    )
    monkeypatch.setattr(
        prepare, "split_patient_ids_stratified",
        lambda labels, seed: {
            patient: ("train", "validation", "test")[i % 3]
            for i, patient in enumerate(sorted(labels))
        },
    )
    monkeypatch.setattr(
        prepare, "split_patient_ids",
        lambda patients, seed: {
            patient: ("train", "validation", "test")[i % 3]
            for i, patient in enumerate(sorted(patients))
        },
    )
    paths = [
        *(prepare.PROCESSED_DIR / name for name in (
            "aligned_beat_aml.tsv.gz", "aligned_tcga_laml.tsv.gz",
            "labels_beat_aml.tsv", "labels_tcga_laml.tsv",
            "jointly_complete_gene_names.txt", "source_train_top2000_genes.txt",
        )),
        *(prepare.SPLITS_DIR / name for name in (
            "beat_aml_splits.tsv", "tcga_laml_splits.tsv",
            "beat_aml_splits_dnmt3a_e7.tsv", "seeds.json",
            "beat_aml_split_counts.tsv",
        )),
        prepare.RESULTS_TABLES / "analysis_set.tsv",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("main sentinel", encoding="utf-8")
    assert prepare.main(analysis_set) == 0
    assert all(path.read_text(encoding="utf-8") == "main sentinel" for path in paths)
    for path in paths:
        if path.name == "jointly_complete_gene_names.txt":
            continue
        assert prepare._output_path(path.parent, path.name, analysis_set).is_file()
    outputs = [
        path for directory in (prepare.PROCESSED_DIR, prepare.SPLITS_DIR, prepare.RESULTS_TABLES)
        for path in directory.iterdir() if path not in paths
    ]
    assert len(outputs) == len(paths) - 1
    assert all(f"_{analysis_set}." in path.name for path in outputs)
