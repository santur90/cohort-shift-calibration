import inspect

import numpy as np
import pandas as pd
import pytest

from cohortshift.adapt import adapt_features, adaptation_weights
from cohortshift.conformal import conformal_quantile, prediction_sets
from cohortshift.data import (
    apl_evidence_samples,
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
    split_patient_ids_stratified,
)
from cohortshift.features import (
    cohort_standardize,
    filter_jointly_complete_genes,
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
    E0_METHODS,
    domain_classifier_auroc,
    domain_classifier_weights,
    per_gene_ks,
    run_e0_shift,
    target_expression_a0,
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
            "ONCOTREE_CODE": "AML",
        },
        "aml_ohsu_2022_2_BA0004": {
            "DISEASE_STAGE_AT_SPECIMEN_COLLECTION": "Other",
            "TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION": "2",
            "ONCOTREE_CODE": "AML",
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
    assert detect_apl_samples(
        clinical, iter(structural_variants), cohort="beat_aml"
    ) == {
        "clinical-apl",
        "sv-pair",
        "sv-event",
    }


def test_apl_evidence_samples_returns_each_evidence_source():
    clinical = {
        "clinical-apl": {
            "CANCER_TYPE_DETAILED": "Acute promyelocytic leukaemia; PML-RARA"
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
    columns, evidence = apl_evidence_samples(
        clinical, structural_variants, cohort="beat_aml"
    )
    assert columns == list(structural_variants[0])
    assert evidence["clinical_cancer_type_detailed"] == {"clinical-apl"}
    assert evidence["sv_partner_genes"] == {"sv-pair"}
    assert evidence["sv_event_info"] == {"sv-event"}
    assert evidence["union"] == {"clinical-apl", "sv-pair", "sv-event"}


def test_apl_detection_requires_nonempty_sv_file_and_matches_tcga_schema(capsys):
    clinical = {"TCGA-AB-2803-03": {"CANCER_TYPE_DETAILED": "Acute myeloid leukemia"}}
    with pytest.raises(ValueError, match="requires data_sv.txt"):
        detect_apl_samples(clinical, None, cohort="tcga_laml")
    with pytest.raises(ValueError, match="non-empty data_sv.txt"):
        detect_apl_samples(clinical, iter(()), cohort="tcga_laml")

    tcga_sv_row = {
        "Sample_Id": "TCGA-AB-2803-03",
        "Site1_Hugo_Symbol": "PML",
        "Site1_Chromosome": "15",
        "Site1_Position": "74033414",
        "Site2_Hugo_Symbol": "RARA",
        "Site2_Chromosome": "17",
        "Site2_Position": "40348316",
        "Site2_Effect_On_Frame": "in-frame",
        "SV_Status": "SOMATIC",
        "Tumor_Split_Read_Count": "2",
        "Tumor_Paired_End_Read_Count": "22",
        "NCBI_Build": "GRCh37",
        "Event_Info": "PML-RARA fusion",
        "Center": "WashU",
    }
    assert detect_apl_samples(
        clinical, [tcga_sv_row], cohort="tcga_laml"
    ) == {"TCGA-AB-2803-03"}
    output = capsys.readouterr().out
    assert "Site2_Effect_On_Frame" in output
    assert "DATA CHECK TCGA-LAML PML-RARA sample counts:" in output
    assert (
        "clinical_diagnosis=0; sv_partner_genes=1; "
        "sv_event_info=1; combined=1"
    ) in output


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


def test_joint_complete_gene_filter_excludes_gene_missing_entirely_in_target():
    source = pd.DataFrame(
        {
            "complete": [1.0, 2.0],
            "target_missing": [3.0, 4.0],
            "partial": [5.0, np.nan],
            "source_only": [6.0, 7.0],
        }
    )
    target = pd.DataFrame(
        {
            "complete": [7.0, 8.0],
            "target_missing": [np.nan, np.nan],
            "partial": [9.0, 10.0],
            "target_only": [11.0, 12.0],
        }
    )
    filtered_source, filtered_target, dropped = filter_jointly_complete_genes(
        source, target
    )
    assert filtered_source.columns.tolist() == ["complete"]
    assert filtered_target.columns.tolist() == ["complete"]
    assert dropped["source"] == 2
    assert dropped["target"] == 2
    assert dropped["total"] == 4
    assert dropped["source_only"] == 1
    assert dropped["target_only"] == 1
    assert dropped["not_shared_total"] == 2
    assert dropped["missing_source"] == 1
    assert dropped["missing_target"] == 1
    assert dropped["missing_total"] == 2
    assert dropped["kept"] == 1
    assert not filtered_source.isna().any().any()
    assert not filtered_target.isna().any().any()


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
    with pytest.raises(ValueError, match="complete measurements"):
        domain_classifier_auroc(source, target, method="A3", seed=7)


def test_patient_splits_stratify_on_source_label():
    labels = {
        **{f"positive-{index}": 1 for index in range(40)},
        **{f"negative-{index}": 0 for index in range(60)},
        **{f"unprofiled-{index}": None for index in range(5)},
    }
    assignments = split_patient_ids_stratified(labels, seed=19)
    assert assignments == split_patient_ids_stratified(labels, seed=19)
    for label in (0, 1):
        counts = {
            split: sum(
                assignments[patient] == split and value == label
                for patient, value in labels.items()
            )
            for split in ("train", "validation", "test")
        }
        assert all(counts[split] > 0 for split in counts)
        assert sum(counts.values()) == sum(value == label for value in labels.values())


def test_e0_reports_a3_weight_diagnostics_without_imputation(tmp_path):
    rng = np.random.default_rng(83)
    source = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    target = pd.DataFrame(rng.normal(loc=0.15, size=(40, 4)), columns=list("abcd"))

    summary, _ = run_e0_shift(
        source,
        target,
        methods=("A3",),
        seed=3,
        pca_output_dir=str(tmp_path),
    )
    row = summary.iloc[0]
    weights = domain_classifier_weights(source.to_numpy(), target.to_numpy())
    expected_ess = np.square(weights.sum()) / np.square(weights).sum()
    assert row["effective_sample_size"] == pytest.approx(expected_ess)
    assert row["weight_min"] == pytest.approx(weights.min())
    assert row["weight_median"] == pytest.approx(np.median(weights))
    assert row["max_weight"] == pytest.approx(weights.max())
    assert row["frac_weights_at_lower_clip"] == pytest.approx(
        np.isclose(weights, 0.1).mean()
    )
    assert row["frac_weights_at_upper_clip"] == pytest.approx(
        np.isclose(weights, 10.0).mean()
    )
    assert row["clip_lower_bound"] == pytest.approx(0.1)
    assert row["clip_upper_bound"] == pytest.approx(10.0)
    max_at_clip = np.isclose(weights.max(), 10.0)
    assert bool(row["max_weight_at_upper_clip"]) == max_at_clip
    expected_degenerate = (
        expected_ess < 0.1 * len(source)
        or max_at_clip
        or np.isclose(weights, 0.1).mean() >= 0.5
        or np.isclose(weights, 10.0).mean() >= 0.5
        or np.allclose(weights, weights[0])
    )
    assert bool(row["degenerate_flag"]) == expected_degenerate
    assert row["degenerate_note"]
    assert "unweighted; unchanged by construction" in row[
        "domain_classifier_auroc_note"
    ]


def test_perfectly_separable_domain_classifier_is_flagged_degenerate(tmp_path):
    source = pd.DataFrame(np.zeros((40, 4)), columns=list("abcd"))
    target = pd.DataFrame(np.ones((40, 4)), columns=list("abcd"))
    summary, _ = run_e0_shift(
        source,
        target,
        methods=("A3",),
        seed=11,
        pca_output_dir=str(tmp_path),
    )
    row = summary.iloc[0]
    assert bool(row["degenerate_flag"])
    assert "all weights equal" in row["degenerate_note"]


def test_all_lower_clip_a3_weights_are_flagged_degenerate(tmp_path, monkeypatch):
    rng = np.random.default_rng(83)
    source = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    target = pd.DataFrame(rng.normal(loc=0.15, size=(40, 4)), columns=list("abcd"))

    def lower_clip_weights(source_features, target_features, *, clip):
        return np.full(len(source_features), clip[0])

    monkeypatch.setattr(
        "cohortshift.shift.domain_classifier_weights", lower_clip_weights
    )
    summary, _ = run_e0_shift(
        source,
        target,
        methods=("A3",),
        seed=3,
        pca_output_dir=str(tmp_path),
    )
    row = summary.iloc[0]
    assert row["weight_min"] == pytest.approx(0.1)
    assert row["weight_median"] == pytest.approx(0.1)
    assert row["frac_weights_at_lower_clip"] == pytest.approx(1.0)
    assert bool(row["degenerate_flag"])
    assert "at least 50% of weights at lower clip" in row["degenerate_note"]


def test_a1_cv_uses_fold_local_standardization_and_preserves_old_metric(tmp_path):
    rng = np.random.default_rng(42)
    source = pd.DataFrame(rng.normal(size=(60, 6)), columns=list("abcdef"))
    target = pd.DataFrame(
        rng.normal(loc=2.0, scale=1.5, size=(60, 6)), columns=list("abcdef")
    )
    summary, _ = run_e0_shift(
        source,
        target,
        methods=("A1",),
        seed=5,
        pca_output_dir=str(tmp_path),
    )
    row = summary.iloc[0]
    assert np.isfinite(row["domain_classifier_auroc_cv"])
    assert np.isfinite(row["domain_classifier_auroc_whole_data_standardized"])
    assert "forces equal class means" in row["domain_classifier_auroc_note"]


def test_a0_is_deterministic_label_free_and_handles_negative_values():
    target = pd.DataFrame(
        {"gene": [-0.5, 0.0, 3.0]},
        index=["negative", "zero", "positive"],
    )
    original = target.copy(deep=True)

    first = target_expression_a0(target)
    second = target_expression_a0(target)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(target, original)
    assert first["gene"].tolist() == pytest.approx([0.0, 0.0, 2.0])
    assert np.isfinite(first.to_numpy()).all()
    assert list(inspect.signature(target_expression_a0).parameters) == ["values"]


def test_e0_includes_a0_and_composite_rows_and_keeps_raw_rows(tmp_path):
    rng = np.random.default_rng(2026)
    source = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    target = pd.DataFrame(rng.normal(loc=1.5, size=(40, 4)), columns=list("abcd"))
    summary, gene_ks = run_e0_shift(
        source,
        target,
        seed=7,
        pca_output_dir=str(tmp_path),
    )
    assert tuple(summary["method"]) == E0_METHODS
    assert set(summary["expression_scale"]) == {
        "raw units",
        "target A0: log2(max(x, 0) + 1)",
    }
    assert set(gene_ks["method"]) == set(E0_METHODS)
    assert gene_ks.loc[gene_ks["method"] == "A0+A3", "p_value"].isna().all()
    for method in E0_METHODS:
        assert (tmp_path / f"e0_pca_{method}.png").is_file()
    assert summary.loc[summary["method"] == "A0+A3", "degenerate_flag"].notna().all()


def test_conformal_quantile_and_prediction_sets():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.25) == 0.3
    assert prediction_sets(np.array([[0.9, 0.1]]), 0.2) == [(0,)]
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]))
