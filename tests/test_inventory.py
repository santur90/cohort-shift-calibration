import csv
import importlib.util
from pathlib import Path


def _write(study: Path, filename: str, text: str) -> None:
    (study / filename).write_text(text, encoding="utf-8")


def test_analysis_set_selection_and_optional_missing_subtype(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "00_inventory.py"
    spec = importlib.util.spec_from_file_location("cohortshift_inventory", script)
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)

    beat = tmp_path / "raw" / "aml_ohsu_2022"
    tcga = tmp_path / "raw" / "laml_tcga_pan_can_atlas_2018"
    beat.mkdir(parents=True)
    tcga.mkdir(parents=True)

    _write(
        beat,
        "data_clinical_sample.txt",
        "#Patient Identifier\n#Other sample attributes\n"
        "#PATIENT_ID\tSAMPLE_ID\tCOHORT\tONCOTREE_CODE\tCANCER_TYPE_DETAILED\t"
        "DISEASE_STAGE_AT_SPECIMEN_COLLECTION\tIS_RELAPSE\tSPECIMEN_GROUPS\t"
        "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION\tNPM1\n"
        "p1\taml_ohsu_2022_1_BA0001\tA\tAML\tAML\tInitial Diagnosis\tFALSE\t"
        "Initial Acute Leukemia Diagnosis\t5\tMutated\n"
        "p1\taml_ohsu_2022_1_BA0002\tA\tAML\tAML\tRelapse\tTRUE\tRelapse\t1\tWildtype\n"
        "p2\taml_ohsu_2022_2_BA0003\tB\tAML\tAML\tOther\tFALSE\tOther\t2\tWildtype\n"
        "p2\taml_ohsu_2022_2_BA0004\tB\tAML\tAML\tOther\tFALSE\tOther\t2\tWildtype\n"
        "p3\taml_ohsu_2022_3_BA0005\tC\tAML\tAML\tOther\tFALSE\tOther\t3\tWildtype\n"
        "p3\taml_ohsu_2022_3_BA0006\tC\tAML\tAML\tOther\tFALSE\tOther\t1\tWildtype\n"
        "p4\taml_ohsu_2022_4_BA0007\tD\tAML\tAML\tInitial Diagnosis\tFALSE\t"
        "Initial Acute Leukemia Diagnosis\t0\tWildtype\n",
    )
    _write(
        beat,
        "data_clinical_patient.txt",
        "#PATIENT_ID\tAGE\np1\t50\np2\t60\np3\t70\np4\t80\n",
    )
    _write(
        beat,
        "data_mrna_seq_rpkm.txt",
        "Hugo_Symbol\tEntrez_Gene_Id\taml_ohsu_2022_1_BA0001\t"
        "aml_ohsu_2022_1_BA0002\taml_ohsu_2022_2_BA0003\t"
        "aml_ohsu_2022_2_BA0004\taml_ohsu_2022_3_BA0005\t"
        "aml_ohsu_2022_3_BA0006\taml_ohsu_2022_4_BA0007\n"
        "NPM1\t1\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t0.7\n",
    )
    _write(
        beat,
        "data_gene_panel_matrix.txt",
        "SAMPLE_ID\tmutations\tgistic\n"
        "aml_ohsu_2022_1_BA0001\tWXS\tNA\n"
        "aml_ohsu_2022_1_BA0002\tWXS\tNA\n"
        "aml_ohsu_2022_2_BA0003\tWXS\tNA\n"
        "aml_ohsu_2022_2_BA0004\tWXS\tNA\n"
        "aml_ohsu_2022_3_BA0005\tWXS\tNA\n"
        "aml_ohsu_2022_3_BA0006\tWXS\tNA\n"
        "aml_ohsu_2022_4_BA0007\tNA\tNA\n",
    )
    _write(
        beat,
        "data_mutations.txt",
        "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\n"
        "DNMT3A\taml_ohsu_2022_1_BA0001\tFrame_Shift_Del\n"
        "NPM1\taml_ohsu_2022_1_BA0002\tMissense_Mutation\n"
        "NPM1\taml_ohsu_2022_2_BA0003\tNonsense_Mutation\n"
        "DNMT3A\taml_ohsu_2022_2_BA0004\tSilent\n"
        "NPM1\taml_ohsu_2022_3_BA0005\tMissense_Mutation\n"
        "NPM1\taml_ohsu_2022_4_BA0007\tMissense_Mutation\n",
    )

    _write(
        tcga,
        "data_clinical_sample.txt",
        "#PATIENT_ID\tSAMPLE_ID\tONCOTREE_CODE\n"
        "t1\tTCGA-AA-0001-03\tAML\n"
        "t2\tTCGA-AA-0002-03\tAML\n"
        "t3\tTCGA-AA-0003-03\tAML\n",
    )
    _write(
        tcga,
        "data_clinical_patient.txt",
        "#PATIENT_ID\tAGE\nt1\t50\nt2\t60\nt3\t70\n",
    )
    _write(
        tcga,
        "data_mrna_seq_v2_rsem.txt",
        "Hugo_Symbol\tEntrez_Gene_Id\tTCGA-AA-0001-03\t"
        "TCGA-AA-0002-03\tTCGA-AA-0003-03\nNPM1\t1\t0.1\t0.2\t0.3\n",
    )
    _write(
        tcga,
        "data_gene_panel_matrix.txt",
        "SAMPLE_ID\tmutations\tgistic\n"
        "TCGA-AA-0001-03\tWXS\tWXS\n"
        "TCGA-AA-0002-03\tWXS\tWXS\n"
        "TCGA-AA-0003-03\tNA\tWXS\n"
        "TCGA-AA-9999-03\tWXS\tWXS\n",
    )
    _write(
        tcga,
        "data_mutations.txt",
        "Hugo_Symbol\tTumor_Sample_Barcode\tVariant_Classification\n"
        "DNMT3A\tTCGA-AA-0001-03\tFrame_Shift_Ins\n"
        "DNMT3A\tTCGA-AA-9999-03\tMissense_Mutation\n"
        "NPM1\tTCGA-AA-0002-03\tSplice_Site\n",
    )
    config = tmp_path / "config.tsv"
    config.write_text(
        "cohort\tstudy_id\trna_file\n"
        "beat_aml\taml_ohsu_2022\tdata_mrna_seq_rpkm.txt\n"
        "tcga_laml\tlaml_tcga_pan_can_atlas_2018\tdata_mrna_seq_v2_rsem.txt\n",
        encoding="utf-8",
    )
    inventory.CONFIG = config
    inventory.RAW_DIR = tmp_path / "raw"
    inventory.OUT = tmp_path / "inventory.tsv"
    inventory.ANALYSIS_OUT = tmp_path / "analysis_set.tsv"

    assert inventory.main() == 0
    printed = capsys.readouterr().out
    assert "DATA CHECK TCGA-LAML SUBTYPE: (column not present)" in printed
    assert (
        "DATA CHECK TCGA-LAML data_clinical_sample.txt columns: "
        "PATIENT_ID | SAMPLE_ID | ONCOTREE_CODE"
    ) in printed
    assert (
        "DATA CHECK TCGA-LAML data_clinical_patient.txt columns: "
        "PATIENT_ID | AGE"
    ) in printed
    first_distinct_check = printed.index("DATA CHECK Beat AML COHORT:")
    for header_line in (
        "DATA CHECK Beat AML data_clinical_sample.txt columns:",
        "DATA CHECK Beat AML data_clinical_patient.txt columns:",
        "DATA CHECK TCGA-LAML data_clinical_sample.txt columns:",
        "DATA CHECK TCGA-LAML data_clinical_patient.txt columns:",
    ):
        assert printed.index(header_line) < first_distinct_check
    assert "DATA CHECK Beat AML COHORT: A | B | C" in printed
    assert "DATA CHECK Beat AML DISEASE_STAGE_AT_SPECIMEN_COLLECTION:" in printed
    assert "DATA CHECK Beat AML eligible stage / IS_RELAPSE / SPECIMEN_GROUPS:" in printed

    with inventory.ANALYSIS_OUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    beat_row = next(row for row in rows if row["cohort"] == "beat_aml")
    tcga_row = next(row for row in rows if row["cohort"] == "tcga_laml")
    assert beat_row["n_samples_with_rna_and_mutation_profiling"] == "6"
    assert beat_row["n_samples_in_analysis_set"] == "1"
    assert beat_row["n_patients"] == "1"
    assert beat_row["selected_sample_ids"] == "aml_ohsu_2022_1_BA0001"
    assert beat_row["npm1_positive_count"] == "0"
    assert beat_row["dnmt3a_positive_count"] == "1"
    assert tcga_row["n_samples_with_rna_and_mutation_profiling"] == "2"
    assert tcga_row["n_patients"] == "2"
    assert tcga_row["target_rna_sample_count"] == "3"
    assert tcga_row["target_rna_dnmt3a_positive_count"] == "1"
    assert tcga_row["dnmt3a_secondary_eligible"] == "false"

    assert inventory.main(check_only=True) == 0
    check_output = capsys.readouterr().out
    assert (
        "DATA CHECK Beat AML: 6 RNA+mutation-profiled candidates, "
        "1 analysis samples"
    ) in check_output
