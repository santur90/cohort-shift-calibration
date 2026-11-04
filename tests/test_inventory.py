import csv
import importlib.util
from pathlib import Path


def test_inventory_counts_only_gene_profiled_mutations(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "00_inventory.py"
    spec = importlib.util.spec_from_file_location("cohortshift_inventory", script)
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)

    study = tmp_path / "raw" / "fixture"
    study.mkdir(parents=True)
    (study / "data_clinical_sample.txt").write_text(
        "#Patient Identifier\n#Other sample attributes\n"
        "#PATIENT_ID\tSAMPLE_ID\tTIMEPOINT\n"
        "p1\ts1\tDiagnosis\np1\ts2\tRelapse\np2\ts3\tDiagnosis\n",
        encoding="utf-8",
    )
    (study / "data_clinical_patient.txt").write_text(
        "#PATIENT_ID\tAGE\np1\t50\np2\t60\n", encoding="utf-8"
    )
    (study / "data_mrna_seq_rpkm.txt").write_text(
        "Hugo_Symbol\tEntrez_Gene_Id\ts1\ts2\nNPM1\t1\t0.1\t0.2\n",
        encoding="utf-8",
    )
    (study / "data_gene_panel_matrix.txt").write_text(
        "SAMPLE_ID\tmutations\tgistic\n"
        "s1\tWXS\tNA\ns2\tWXS\tNA\ns3\tNA\tNA\n",
        encoding="utf-8",
    )
    (study / "data_mutations.txt").write_text(
        "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\n"
        "NPM1\ts1\tMissense_Mutation\n"
        "NPM1\ts3\tMissense_Mutation\n"
        "DNMT3A\ts1\tSilent\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.tsv"
    config.write_text(
        "cohort\tstudy_id\trna_file\nfixture\tfixture\tdata_mrna_seq_rpkm.txt\n",
        encoding="utf-8",
    )
    output = tmp_path / "inventory.tsv"
    inventory.CONFIG = config
    inventory.RAW_DIR = tmp_path / "raw"
    inventory.OUT = output

    assert inventory.main() == 0
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    summary = next(row for row in rows if row["record_type"] == "cohort_summary")
    npm1 = next(
        row for row in rows if row["record_type"] == "gene_panel" and row["gene"] == "NPM1"
    )
    dnmt3a = next(
        row
        for row in rows
        if row["record_type"] == "gene_panel" and row["gene"] == "DNMT3A"
    )
    sample_file = next(
        row
        for row in rows
        if row["record_type"] == "file"
        and row["file"] == "data_clinical_sample.txt"
    )
    assert (summary["sample_count"], summary["patient_count"], summary["rna_sample_count"]) == (
        "3",
        "2",
        "2",
    )
    assert npm1["panel_covered"] == "true"
    assert npm1["profiled_sample_count"] == "2"
    assert npm1["positive_sample_count"] == "1"
    assert dnmt3a["positive_sample_count"] == "0"
    assert "TIMEPOINT" in sample_file["diagnosis_column_candidates"]
    assert "TIMEPOINT" in sample_file["relapse_column_candidates"]
