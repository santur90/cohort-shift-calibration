import inspect

import numpy as np
import pandas as pd
import pytest

from cohortshift.adapt import adapt_features, adaptation_weights
from cohortshift.conformal import conformal_quantile, prediction_sets
from cohortshift.data import (
    detect_apl_samples,
    find_header,
    iter_tsv,
    load_expression_matrix,
    matrix_sample_columns,
    mutation_profile_samples,
    parse_patient_id,
    sample_id_format,
    select_one_sample_per_patient,
    split_patient_ids,
)
from cohortshift.features import (
    cohort_standardize,
    select_top_variance_genes,
    within_sample_rank,
)
from cohortshift.labels import (
    NON_SILENT_VARIANT_CLASSIFICATIONS,
    diagnosis_relapse_candidates,
    is_non_silent_variant,
    labels_from_maf_rows,
    mutation_labels,
)
from cohortshift.shift import (
    domain_classifier_auroc,
    domain_classifier_weights,
    per_gene_ks,
)


def test_cbioportal_clinical_metadata_header(tmp_path):
    path = tmp_path / "clinical.txt"
    path.write_text(
        "#Patient Identifier\n#Other patient attributes\n"
        "#PATIENT_ID\tSAMPLE_ID\tSAMPLE_TYPE\np1\tTCGA-AA-0001-01A\tPrimary Tumor\n",
        encoding="utf-8",
    )
    assert find_header(path) == ["PATIENT_ID", "SAMPLE_ID", "SAMPLE_TYPE"]
    assert list(iter_tsv(path))[0]["SAMPLE_ID"] == "TCGA-AA-0001-01A"


def test_matrix_columns_and_id_pattern():
    samples = matrix_sample_columns(["Hugo_Symbol", "Entrez_Gene_Id", "TCGA-AA-0001-01A"])
    assert samples == ["TCGA-AA-0001-01A"]
    assert "pattern=XXXX-XX-####-##X" in sample_id_format(samples)
    assert matrix_sample_columns(["SAMPLE_ID", "mutations", "gistic"]) == []


def test_expression_load_keeps_highest_mean_duplicate_hugo_row(tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text(
        "Hugo_Symbol\tEntrez_Gene_Id\ts1\ts2\n"
        "NPM1\t1\t1\t2\n"
        "NPM1\t1\t3\t4\n"
        "DNMT3A\t2\t2\t2\n",
        encoding="utf-8",
    )
    expression = load_expression_matrix(path)
    assert expression.index.tolist() == ["s1", "s2"]
    assert expression.columns.tolist() == ["DNMT3A", "NPM1"]
    assert expression.loc["s1", "NPM1"] == 3


def test_patient_id_parsing_and_registered_analysis_set_selection():
    assert parse_patient_id("beat_aml", "aml_ohsu_2022_2476_BA2102") == (
        "aml_ohsu_2022_2476"
    )
    assert parse_patient_id("tcga_laml", "TCGA-AB-2989-03") == "TCGA-AB-2989"
    samples = {
        "aml_ohsu_2022_1_BA0001",
        "aml_ohsu_2022_1_BA0002",
        "aml_ohsu_2022_2_BA0003",
        "aml_ohsu_2022_2_BA0004",
        "aml_ohsu_2022_3_BA0005",
        "aml_ohsu_2022_3_BA0008",
        "aml_ohsu_2022_3_BA0009",
        "aml_ohsu_2022_4_BA0006",
        "aml_ohsu_2022_5_BA0007",
    }
    clinical = {
        "aml_ohsu_2022_1_BA0001": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Relapse",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "1",
            "ONCOTREE_CODE": "AML",
        },
        "aml_ohsu_2022_1_BA0002": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "5",
            "ONCOTREE_CODE": "AML",
        },
        "aml_ohsu_2022_2_BA0003": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Other",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "2",
        },
        "aml_ohsu_2022_2_BA0004": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Other",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "2",
        },
        "aml_ohsu_2022_3_BA0005": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "2",
            "ONCOTREE_CODE": "AML",
        },
        "aml_ohsu_2022_3_BA0008": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "1",
            "ONCOTREE_CODE": "AML",
        },
        "aml_ohsu_2022_3_BA0009": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "1",
            "ONCOTREE_CODE": "AML",
        },
        "aml_ohsu_2022_4_BA0006": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "2",
            "ONCOTREE_CODE": "ALAL",
        },
        "aml_ohsu_2022_5_BA0007": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Initial Diagnosis",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "2",
            "ONCOTREE_CODE": "AML",
        },
    }
    main = select_one_sample_per_patient("beat_aml", samples, clinical)
    assert main == {
        "aml_ohsu_2022_1_BA0002",
        "aml_ohsu_2022_3_BA0008",
        "aml_ohsu_2022_4_BA0006",
        "aml_ohsu_2022_5_BA0007",
    }
    assert all(
        clinical[sample]["DISEASE_STAGE_AT_SPECIMEN_COLLECTION"]
        == "Initial Diagnosis"
        for sample in main
    )
    assert len({parse_patient_id("beat_aml", sample) for sample in main}) == len(main)
    e6b = select_one_sample_per_patient(
        "beat_aml", samples, clinical, analysis_set="e6b"
    )
    assert "aml_ohsu_2022_2_BA0003" in e6b
    assert "aml_ohsu_2022_2_BA0004" not in e6b
    assert not ({"aml_ohsu_2022_2_BA0003", "aml_ohsu_2022_2_BA0004"} & main)
    assert "aml_ohsu_2022_3_BA0009" not in main
    e6a = select_one_sample_per_patient(
        "beat_aml",
        samples,
        clinical,
        analysis_set="e6a",
        apl_samples={"aml_ohsu_2022_5_BA0007"},
    )
    assert e6a == {
        "aml_ohsu_2022_1_BA0002",
        "aml_ohsu_2022_3_BA0008",
    }


def test_apl_detection_uses_clinical_and_structural_variant_evidence():
    clinical = {
        "clinical-apl": {
            "CANCER_TYPE_DETAILED":
                "Acute promyelocytic leukaemia with t(15;17); PML-RARA"
        },
        "clinical-normal": {"CANCER_TYPE_DETAILED": "Acute myeloid leukaemia"},
    }
    structural_variants = [
        {
            "Sample_Id": "sv-pair",
            "Site1_Hugo_Symbol": "PML",
            "Site2_Hugo_Symbol": "RARA",
            "Event_Info": "",
        },
        {
            "Sample_Id": "sv-event",
            "Site1_Hugo_Symbol": "PML",
            "Site2_Hugo_Symbol": "OTHER",
            "Event_Info": "PML-RARA Fusion",
        },
    ]
    assert detect_apl_samples(clinical, iter(structural_variants)) == {
        "clinical-apl",
        "sv-pair",
        "sv-event",
    }


def test_e6a_applies_aml_and_apl_filters_to_tcga():
    clinical = {
        "TCGA-AA-0001-03": {"ONCOTREE_CODE": "AML"},
        "TCGA-AA-0002-03": {"ONCOTREE_CODE": "AML"},
        "TCGA-AA-0003-03": {"ONCOTREE_CODE": "ALAL"},
    }
    assert select_one_sample_per_patient(
        "tcga_laml",
        set(clinical),
        clinical,
        analysis_set="e6a",
        apl_samples={"TCGA-AA-0002-03"},
    ) == {"TCGA-AA-0001-03"}


def test_patient_splits_are_disjoint_and_repeatable():
    assignments = split_patient_ids({f"patient-{index}" for index in range(30)}, seed=42)
    assert assignments == split_patient_ids(
        {f"patient-{index}" for index in range(30)}, seed=42
    )
    patients_by_split = {
        split: {patient for patient, assigned in assignments.items() if assigned == split}
        for split in {"train", "validation", "test"}
    }
    assert not (patients_by_split["train"] & patients_by_split["validation"])
    assert not (patients_by_split["train"] & patients_by_split["test"])
    assert not (patients_by_split["validation"] & patients_by_split["test"])
    split_sizes = {
        split: sum(assigned == split for assigned in assignments.values())
        for split in ("train", "validation", "test")
    }
    assert split_sizes == {"train": 18, "validation": 6, "test": 6}
    main_set_splits = split_patient_ids(
        {f"patient-{index}" for index in range(426)}, seed=42
    )
    assert {
        split: sum(assigned == split for assigned in main_set_splits.values())
        for split in ("train", "validation", "test")
    } == {"train": 256, "validation": 85, "test": 85}


def test_gene_panel_matrix_uses_sample_rows_and_excludes_unknown_panels(tmp_path):
    path = tmp_path / "panel.txt"
    path.write_text(
        "SAMPLE_ID\tmutations\tgistic\n"
        "s1\tWXS\tWXS\n"
        "s2\tNA\tWXS\n"
        "s3\tTARGETED_42\tNA\n",
        encoding="utf-8",
    )
    profiled, panel_names = mutation_profile_samples(path, {"NPM1", "DNMT3A"})
    assert profiled["NPM1"] == {"s1"}
    assert profiled["DNMT3A"] == {"s1"}
    assert panel_names["NPM1"] == {"WXS"}


def test_diagnosis_relapse_candidates_does_not_assume_column_name():
    diagnosis, relapse = diagnosis_relapse_candidates(
        {
            "sample": {
                "TIMEPOINT": ["Diagnosis", "Relapse"],
                "SAMPLE_CLASS": ["Primary", "Recurrent"],
            }
        }
    )
    assert "sample:TIMEPOINT" in diagnosis
    assert "sample:TIMEPOINT" in relapse
    assert "sample:SAMPLE_CLASS" in relapse


def test_unprofiled_mutations_are_missing_not_negative():
    assert mutation_labels({"s1"}, {"s1", "s2"}, {"s1", "s2", "s3"}) == {
        "s1": 1,
        "s2": 0,
        "s3": None,
    }


def test_maf_label_builder_leaves_unprofiled_samples_missing():
    labels = labels_from_maf_rows(
        [
            {
                "Hugo_Symbol": "NPM1",
                "Tumor_Sample_Barcode": "s1",
                "Variant_Classification": "Missense_Mutation",
            },
            {
                "Hugo_Symbol": "NPM1",
                "Tumor_Sample_Barcode": "s3",
                "Variant_Classification": "Nonsense_Mutation",
            },
        ],
        {"NPM1"},
        {"s1", "s2", "s3"},
        {"NPM1": {"s1", "s2"}},
    )
    assert labels.loc["s1", "NPM1"] == 1
    assert labels.loc["s2", "NPM1"] == 0
    assert pd.isna(labels.loc["s3", "NPM1"])



def test_non_silent_definition_is_exact_registered_set():
    assert NON_SILENT_VARIANT_CLASSIFICATIONS == {
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
    assert is_non_silent_variant("Missense_Mutation")
    assert not is_non_silent_variant("Silent")
    assert not is_non_silent_variant("missense_mutation")


def test_feature_adaptation_and_constant_scale():
    values = pd.DataFrame([[2.0, 5.0], [4.0, 5.0]], columns=["a", "b"])
    standardized, _, scale = cohort_standardize(values)
    assert np.isfinite(standardized.to_numpy()).all()
    assert scale["b"] == 1
    ranked = within_sample_rank(values)
    assert ranked.iloc[0].tolist() == [0.5, 1.0]
    source, target = adapt_features("A2", values, values)
    pd.testing.assert_frame_equal(source, target)
    assert select_top_variance_genes(values, n_genes=1) == ["a"]


def test_adaptation_interfaces_do_not_accept_target_labels():
    assert "target_labels" not in inspect.signature(adapt_features).parameters
    assert "target_labels" not in inspect.signature(adaptation_weights).parameters


def test_domain_weights_are_clipped():
    weights = domain_classifier_weights(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([[3.0], [4.0]]),
        clip=(0.2, 2.0),
    )
    assert np.all((weights >= 0.2) & (weights <= 2.0))


def test_domain_classifier_auroc_is_chance_for_same_distribution():
    rng = np.random.default_rng(23)
    source = pd.DataFrame(rng.normal(size=(120, 8)), columns=list("abcdefgh"))
    target = pd.DataFrame(rng.normal(size=(120, 8)), columns=list("abcdefgh"))
    auroc = domain_classifier_auroc(source, target, seed=7)
    assert 0.4 <= auroc <= 0.6
    source.iloc[0, 0] = np.nan
    target.iloc[1, 1] = np.nan
    weighted_auroc = domain_classifier_auroc(source, target, method="A3", seed=7)
    weighted_ks = per_gene_ks(source, target, method="A3")
    assert 0.4 <= weighted_auroc <= 0.6
    assert len(weighted_ks) == source.shape[1]
    assert weighted_ks["p_value"].isna().all()


def test_conformal_quantile_and_prediction_sets():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.25) == 0.3
    assert prediction_sets(np.array([[0.9, 0.1]]), 0.2) == [(0,)]
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]))
