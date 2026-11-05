import csv
import importlib.util
from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_read_only_selection_check_tables_and_counts(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "01_check_selection.py"
    spec = importlib.util.spec_from_file_location("cohortshift_selection_check", script)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    beat = tmp_path / "aml_ohsu_2022"
    beat.mkdir()
    samples = [
        ("aml_ohsu_2022_1_BA0001", "AML", "Initial Diagnosis", "5"),
        ("aml_ohsu_2022_1_BA0002", "MDS", "Relapse", "1"),
        ("aml_ohsu_2022_2_BA0003", "ALAL", "Relapse", "5"),
        ("aml_ohsu_2022_2_BA0004", "MPN", "Unknown", "1"),
        ("aml_ohsu_2022_3_BA0005", "MDS/MPN", "Unknown", "3"),
        ("aml_ohsu_2022_4_BA0006", "AML", "Initial Diagnosis", "0"),
    ]
    _write(
        beat / "data_mrna_seq_rpkm.txt",
        "Hugo_Symbol\tEntrez_Gene_Id\t"
        + "\t".join(sample[0] for sample in samples)
        + "\nNPM1\t1\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\n",
    )
    clinical_lines = [
        "#PATIENT_ID\tSAMPLE_ID\tONCOTREE_CODE\t"
        "DISEASE_STAGE_AT_SPECIMEN_COLLECTION\t"
        "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION"
    ]
    for sample_id, code, stage, time in samples:
        patient_id = sample_id.rsplit("_BA", 1)[0]
        clinical_lines.append(
            f"{patient_id}\t{sample_id}\t{code}\t{stage}\t{time}"
        )
    _write(beat / "data_clinical_sample.txt", "\n".join(clinical_lines) + "\n")
    _write(
        beat / "data_gene_panel_matrix.txt",
        "SAMPLE_ID\tmutations\n"
        + "".join(f"{sample[0]}\tWXS\n" for sample in samples),
    )
    _write(
        beat / "data_mutations.txt",
        "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\n"
        "NPM1\taml_ohsu_2022_1_BA0001\tMissense_Mutation\n"
        "DNMT3A\taml_ohsu_2022_1_BA0001\tFrame_Shift_Del\n"
        "NPM1\taml_ohsu_2022_1_BA0002\tMissense_Mutation\n"
        "NPM1\taml_ohsu_2022_2_BA0004\tSilent\n"
        "DNMT3A\taml_ohsu_2022_2_BA0004\tIn_Frame_Ins\n"
        "NPM1\taml_ohsu_2022_3_BA0005\tSplice_Site\n",
    )
    checker.BEAT_AML_DIR = beat
    checker.OUTPUT = tmp_path / "results" / "tables" / "selection_check.tsv"
    checker.OUTPUT.parent.mkdir(parents=True)

    assert checker.main() == 0
    output = capsys.readouterr().out
    assert "fallback patients with no Initial Diagnosis candidate: 2" in output
    assert "all_selected NPM1: N=4, positives=2, prevalence=0.500000" in output
    assert "initial_diagnosis_AML DNMT3A: N=2, positives=1, prevalence=0.500000" in output

    with checker.OUTPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    candidate_unknown = next(
        row
        for row in rows
        if row["section"] == "candidate_stage_counts" and row["stage"] == "Unknown"
    )
    selected_initial = next(
        row
        for row in rows
        if row["section"] == "selected_stage_counts"
        and row["stage"] == "Initial Diagnosis"
    )
    selected_mpn = next(
        row
        for row in rows
        if row["section"] == "selected_oncotree_counts"
        and row["oncotree_code"] == "MPN"
    )
    initial_aml_npm1 = next(
        row
        for row in rows
        if row["section"] == "mutation_counts"
        and row["subset"] == "initial_diagnosis_AML"
        and row["gene"] == "NPM1"
    )
    assert (candidate_unknown["n"], candidate_unknown["patients"]) == ("2", "2")
    assert selected_initial["n"] == "2"
    assert selected_mpn["n"] == "1"
    assert (
        initial_aml_npm1["n"],
        initial_aml_npm1["positives"],
        float(initial_aml_npm1["prevalence"]),
    ) == ("2", "1", 0.5)
    assert list(checker.OUTPUT.parent.iterdir()) == [checker.OUTPUT]
