import inspect
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.container import ErrorbarContainer
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cohortshift.adapt import adapt_features, adaptation_weights
from cohortshift.conformal import (
    conformal_coverage_metrics,
    conformal_quantile,
    prediction_sets,
)
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
from cohortshift.experiments import (
    E3_METHODS,
    _append_repeat_summaries,
    _assemble_cross_cohort_tables,
    _bootstrap_metric_intervals,
    predict_adapted_probabilities,
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
from cohortshift.plots import save_coverage_plot, save_target_reliability_plot
from cohortshift.shift import (
    E0_METHODS,
    domain_classifier_auroc,
    domain_classifier_weights,
    run_e0_shift,
    target_expression_a0,
)
from cohortshift.source_experiments import (
    _apply_calibrator,
    _fit_calibrator,
    expected_calibration_error,
    run_source_experiments,
    select_repeat_genes,
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
        for split in ("train", "validation", "test")
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
    metrics = conformal_coverage_metrics(
        np.array([0, 1]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        0.25,
    )
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["positive_class_coverage"] == pytest.approx(1.0)
    assert metrics["negative_class_coverage"] == pytest.approx(1.0)
    assert metrics["mean_prediction_set_size"] == pytest.approx(1.0)
    assert metrics["empty_set_fraction"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]))


def test_split_conformal_has_exchangeable_synthetic_coverage():
    rng = np.random.default_rng(591)
    all_probabilities = rng.uniform(0.02, 0.98, 4000)
    all_labels = rng.binomial(1, all_probabilities)
    calibration_probabilities = all_probabilities[:2000]
    calibration_labels = all_labels[:2000]
    target_probabilities = all_probabilities[2000:]
    target_labels = all_labels[2000:]
    calibration_scores = 1 - np.where(
        calibration_labels == 1,
        calibration_probabilities,
        1 - calibration_probabilities,
    )
    threshold = conformal_quantile(calibration_scores, alpha=0.1)
    metrics = conformal_coverage_metrics(
        target_labels, target_probabilities, threshold
    )
    assert metrics["coverage"] >= 0.9 - 0.04


def test_equal_count_ece_separates_calibrated_and_overconfident_predictions():
    group_probabilities = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 20)
    labels = np.concatenate(
        [
            np.r_[np.ones(int(probability * 20)), np.zeros(20 - int(probability * 20))]
            for probability in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
    ).astype(int)
    calibrated_ece, calibrated_bins = expected_calibration_error(
        labels, group_probabilities
    )
    overconfident_probabilities = np.repeat([0.01, 0.05, 0.95, 0.99, 0.999], 20)
    overconfident_ece, overconfident_bins = expected_calibration_error(
        labels, overconfident_probabilities
    )
    assert calibrated_ece == pytest.approx(0.0)
    assert overconfident_ece > calibrated_ece + 0.2
    assert [row["bin_count"] for row in calibrated_bins] == [20] * 5
    assert [row["bin_count"] for row in overconfident_bins] == [20] * 5


def test_train_only_gene_selection_ignores_test_expression_and_labels():
    rng = np.random.default_rng(99)
    sample_ids = [f"sample-{index}" for index in range(40)]
    expression = pd.DataFrame(
        rng.normal(size=(40, 8)), index=sample_ids, columns=[f"g{i}" for i in range(8)]
    )
    training_ids = sample_ids[:24]
    original_genes = select_repeat_genes(expression, training_ids, n_genes=5)

    changed_expression = expression.copy()
    changed_expression.loc[sample_ids[24:]] = 1e9
    changed_test_labels = pd.Series(
        np.tile([0, 1], 8), index=sample_ids[24:]
    )
    assert changed_test_labels.eq(1).any()
    assert (
        select_repeat_genes(changed_expression, training_ids, n_genes=5)
        == original_genes
    )
    assert "labels" not in inspect.signature(select_repeat_genes).parameters


def test_source_experiment_functions_accept_no_target_data():
    for function in (run_source_experiments, select_repeat_genes):
        assert all(
            "target" not in parameter.casefold()
            for parameter in inspect.signature(function).parameters
        )


def test_source_experiment_tables_are_bit_identical_for_same_seed():
    rng = np.random.default_rng(7123)
    sample_ids = [f"source-{index:03d}" for index in range(70)]
    labels = pd.Series(np.tile([0, 1], 35), index=sample_ids, name="NPM1")
    values = rng.normal(size=(70, 10))
    values[:, 0] += labels.to_numpy() * 1.5
    expression = pd.DataFrame(
        values, index=sample_ids, columns=[f"gene-{index}" for index in range(10)]
    )
    patient_by_sample = {sample: f"patient-{index:03d}" for index, sample in enumerate(sample_ids)}

    first = run_source_experiments(
        expression,
        labels,
        patient_by_sample,
        seeds=(7201,),
        n_genes=6,
        registered_main_split={
            sample: (
                "train"
                if index < 42
                else "validation"
                if index < 56
                else "test"
            )
            for index, sample in enumerate(sample_ids)
        },
    )
    second = run_source_experiments(
        expression,
        labels,
        patient_by_sample,
        seeds=(7201,),
        n_genes=6,
        registered_main_split={
            sample: (
                "train"
                if index < 42
                else "validation"
                if index < 56
                else "test"
            )
            for index, sample in enumerate(sample_ids)
        },
    )
    for name in ("e1", "e2", "reliability", "splits", "selected_genes"):
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)
        first_buffer = StringIO()
        second_buffer = StringIO()
        first[name].to_csv(first_buffer, sep="\t", index=False)
        second[name].to_csv(second_buffer, sep="\t", index=False)
        assert first_buffer.getvalue() == second_buffer.getvalue()
    repeated_splits = first["splits"].loc[
        first["splits"]["split_set"] == "20_repeats"
    ]
    split_counts = repeated_splits.groupby("split")["patient_id"].nunique()
    assert split_counts.to_dict() == {"calibration": 14, "test": 14, "train": 42}
    registered_counts = (
        first["splits"]
        .loc[first["splits"]["split_set"] == "registered_main_split"]
        .groupby("split")["patient_id"]
        .nunique()
    )
    assert registered_counts.to_dict() == {"test": 14, "train": 42, "validation": 14}
    assert len(first["reliability"]) == 2 * 3 * 5 * 2
    main_rows = first["e2"].loc[
        first["e2"]["split_set"] == "registered_main_split"
    ]
    assert len(main_rows) == 2 * 3
    assert not main_rows["included_in_20_repeats"].any()
    assert set(main_rows["row_type"]) == {"registered_main_split"}
    assert {"mean", "sd", "median", "q1", "q3"}.issubset(
        set(first["e2"]["row_type"])
    )
    assert {
        "slope_gt_5_repeats",
        "slope_fit_not_converged_repeats",
        "slope_gt_5_or_fit_not_converged_repeats",
        "frac_pred_at_clip_bounds",
    }.issubset(first["e2"].columns)
    paired = first["e2"].loc[
        first["e2"]["row_type"].str.startswith("paired_difference")
    ]
    assert len(paired) == 2 * 2 * 3


def test_platt_calibrator_recovers_known_logit_slope():
    rng = np.random.default_rng(108)
    base_logits = rng.uniform(-3.0, 3.0, size=20000)
    base_probabilities = expit(base_logits)
    labels = rng.binomial(1, expit(-0.35 + 1.65 * base_logits))

    calibrator = _fit_calibrator("Platt", base_probabilities, labels)

    assert calibrator is not None
    assert calibrator.penalty is None
    assert calibrator.solver == "L-BFGS-B"
    assert calibrator.max_iter == 2000
    assert calibrator.coef_[0, 0] == pytest.approx(1.65, abs=0.06)


def test_platt_smoothed_targets_avoid_extremes_and_preserve_rank():
    probabilities = np.r_[np.linspace(0.001, 0.05, 80), np.linspace(0.95, 0.999, 80)]
    labels = np.r_[np.zeros(80, dtype=int), np.ones(80, dtype=int)]
    calibrator = _fit_calibrator("Platt", probabilities, labels)
    assert calibrator is not None
    calibrated = _apply_calibrator("Platt", calibrator, probabilities)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) > 0)
    assert roc_auc_score(labels, probabilities) == roc_auc_score(
        labels, calibrated
    )


def test_e4_a1_uses_source_train_statistics_and_avoids_second_scaler():
    source_train = pd.DataFrame(
        {"a": [0.0, 2.0, 4.0, 6.0], "b": [10.0, 12.0, 14.0, 16.0]}
    )
    source_test = pd.DataFrame({"a": [100.0], "b": [110.0]})
    target = pd.DataFrame({"a": [20.0, 24.0], "b": [-5.0, 5.0]})
    train_z, target_z = adapt_features(
        "A1", source_train, target, source_train_features=source_train
    )
    test_z, _ = adapt_features(
        "A1", source_test, target, source_train_features=source_train
    )
    assert train_z.mean().to_numpy() == pytest.approx([0.0, 0.0])
    assert target_z.mean().to_numpy() == pytest.approx([0.0, 0.0])
    expected_test_a = (100.0 - source_train["a"].mean()) / source_train["a"].std(ddof=0)
    assert test_z.iloc[0, 0] == pytest.approx(expected_test_a)


def test_e3_predictions_are_label_free_for_target_and_seed_stable():
    rng = np.random.default_rng(871)
    source_train = pd.DataFrame(rng.normal(size=(60, 6)), columns=list("abcdef"))
    train_labels = np.tile([0, 1], 30)
    source_calibration = pd.DataFrame(rng.normal(size=(20, 6)), columns=list("abcdef"))
    calibration_labels = np.tile([0, 1], 10)
    source_test = pd.DataFrame(rng.normal(size=(20, 6)), columns=list("abcdef"))
    target = pd.DataFrame(rng.normal(loc=0.5, size=(18, 6)), columns=list("abcdef"))
    result = predict_adapted_probabilities(
        source_train,
        train_labels,
        source_calibration,
        calibration_labels,
        source_test,
        target,
        method="A0+A1",
        seed=229,
    )
    result_again = predict_adapted_probabilities(
        source_train,
        train_labels,
        source_calibration,
        calibration_labels,
        source_test,
        target,
        method="A0+A1",
        seed=229,
    )
    shuffled_target_labels = rng.permutation(np.tile([0, 1], 9))
    assert shuffled_target_labels.size == len(target)
    assert "target_labels" not in inspect.signature(
        predict_adapted_probabilities
    ).parameters
    pd.testing.assert_frame_equal(
        result["adapted_target_features"], result_again["adapted_target_features"]
    )
    for model_name in ("logistic", "hist_gradient_boosting"):
        original_model = result["predictions"][model_name]
        repeat_model = result_again["predictions"][model_name]
        np.testing.assert_array_equal(
            original_model["target_probabilities"],
            repeat_model["target_probabilities"],
        )
    assert isinstance(
        result["predictions"]["logistic"]["estimator"], LogisticRegression
    )
    assert "standardscaler" not in result["predictions"]["logistic"][
        "estimator"
    ].get_params()


def test_degenerate_a3_prediction_reports_failed_status(monkeypatch):
    from cohortshift import experiments

    rng = np.random.default_rng(128)
    source = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    target = pd.DataFrame(rng.normal(size=(12, 5)), columns=list("abcde"))
    labels = np.tile([0, 1], 30)
    calibration = source.iloc[:20].copy()
    calibration_labels = labels[:20]
    monkeypatch.setattr(
        experiments,
        "adaptation_weights",
        lambda source_values, target_values: np.full(len(source_values), 0.1),
    )

    result = predict_adapted_probabilities(
        source,
        labels,
        calibration,
        calibration_labels,
        source.iloc[20:40],
        target,
        method="A0+A3",
        seed=402,
    )

    assert result["failed"] is True
    assert result["status"] == "failed"
    assert result["weight_diagnostics"]["degenerate_flag"] is True


def test_raw_a3_uses_normalized_importance_weights(monkeypatch):
    from cohortshift import experiments

    rng = np.random.default_rng(128)
    source = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    target = pd.DataFrame(rng.normal(size=(12, 5)), columns=list("abcde"))
    labels = np.tile([0, 1], 30)
    captured = {}
    unnormalized_weights = np.linspace(0.5, 1.5, len(source))

    monkeypatch.setattr(
        experiments,
        "adaptation_weights",
        lambda source_values, target_values: unnormalized_weights.copy(),
    )

    def capture_fit(train_x, train_labels, **kwargs):
        captured["weights"] = kwargs["sample_weights"]
        return {}

    monkeypatch.setattr(experiments, "_tuned_estimators", capture_fit)
    result = predict_adapted_probabilities(
        source,
        labels,
        source.iloc[:20],
        labels[:20],
        source.iloc[20:40],
        target,
        method="A3",
        seed=402,
    )

    np.testing.assert_allclose(captured["weights"].mean(), 1.0)
    np.testing.assert_allclose(
        captured["weights"], unnormalized_weights / unnormalized_weights.mean()
    )
    assert result["failed"] is False
    assert result["weight_diagnostics"]["degenerate_flag"] is False


def test_cross_cohort_tables_are_deterministic_and_include_registered_outputs(
    monkeypatch,
):
    from cohortshift import experiments

    bootstrap_calls = []

    def fake_metric_intervals(labels, probabilities, patient_ids, *, seed):
        bootstrap_calls.append(seed)
        intervals = {}
        for metric in (
            "auroc",
            "auprc",
            "brier",
            "ece",
            "calibration_intercept",
            "calibration_slope",
        ):
            intervals[f"{metric}_ci_lower"] = 0.1
            intervals[f"{metric}_ci_upper"] = 0.9
        intervals["bootstrap_slope_fit_failed_or_gt5_fraction"] = 0.0
        intervals["bootstrap_n_resamples"] = 2000
        return intervals

    def fake_coverage_intervals(
        labels, probabilities, threshold, patient_ids, *, seed
    ):
        return {
            f"target_{metric}_ci_{bound}": 0.5
            for metric in (
                "coverage",
                "positive_class_coverage",
                "negative_class_coverage",
                "mean_prediction_set_size",
                "empty_set_fraction",
            )
            for bound in ("lower", "upper")
        }

    monkeypatch.setattr(
        experiments, "_bootstrap_metric_intervals", fake_metric_intervals
    )
    monkeypatch.setattr(
        experiments, "_bootstrap_coverage_intervals", fake_coverage_intervals
    )
    rng = np.random.default_rng(584)
    calibration_labels = np.tile([0, 1], 30)
    calibration_probabilities = np.clip(
        expit((calibration_labels - 0.5) * 1.5 + rng.normal(size=60)),
        0.02,
        0.98,
    )
    source_test_labels = np.tile([0, 1], 30)
    source_test_probabilities = np.clip(
        expit((source_test_labels - 0.5) * 1.4 + rng.normal(size=60)),
        0.02,
        0.98,
    )
    target_labels = np.tile([0, 1], 30)
    target_patients = [f"target-{index}" for index in range(len(target_labels))]
    main_entries = []
    repeat_entries = []
    for method_index, method in enumerate(E3_METHODS):
        predictions = {}
        for model_index, model in enumerate(("logistic", "hist_gradient_boosting")):
            target_probabilities = np.clip(
                expit(
                    (target_labels - 0.5) * 1.2
                    + rng.normal(scale=1.0, size=len(target_labels))
                    + (method_index + model_index) * 0.03
                ),
                0.02,
                0.98,
            )
            calibrators = {
                kind: _fit_calibrator(
                    kind, calibration_probabilities, calibration_labels
                )
                for kind in ("none", "Platt", "isotonic")
            }
            predictions[model] = {
                "hyperparameters": {"synthetic": True},
                "calibrators": calibrators,
                "source_calibration_probabilities": calibration_probabilities,
                "source_test_probabilities": source_test_probabilities,
                "target_probabilities": target_probabilities,
            }
        common = {
            "method": method,
            "outcome": "NPM1",
            "failed": False,
            "status": "ready",
            "failure_reason": "",
            "diagnostics": {
                "domain_classifier_auroc_cv": 0.6,
                "mean_gene_ks": 0.2,
            },
            "predictions": predictions,
            "source_calibration_labels": calibration_labels,
            "source_test_labels": source_test_labels,
        }
        main_entries.append(
            {
                **common,
                "split_set": "registered_main_split",
                "repeat": "registered_main_split",
            }
        )
        repeat_entries.append(
            {**common, "split_set": "20_repeats", "repeat": 1}
        )
    prediction_bundle = {
        "outcome": "NPM1",
        "target_samples": [f"target-sample-{index}" for index in range(60)],
        "main": main_entries,
        "repeats": repeat_entries,
    }

    first = _assemble_cross_cohort_tables(
        prediction_bundle,
        target_labels,
        target_patients,
        np.arange(len(target_labels)),
        bootstrap_seed=117,
    )
    second = _assemble_cross_cohort_tables(
        prediction_bundle,
        target_labels,
        target_patients,
        np.arange(len(target_labels)),
        bootstrap_seed=117,
    )

    for table_name in ("cross_cohort", "repeats", "conformal", "reliability"):
        pd.testing.assert_frame_equal(first[table_name], second[table_name])
    main_table = first["cross_cohort"]
    assert len(main_table) == len(E3_METHODS) * 2 * 3
    assert set(main_table["method"]) == set(E3_METHODS)
    assert "frac_pred_at_clip_bounds" in main_table
    assert "target_bootstrap_slope_fit_failed_or_gt5_fraction" in main_table
    assert "target_brier_difference_a0_a1_minus_a0" in main_table
    for metric in (
        "auroc",
        "auprc",
        "brier",
        "ece",
    ):
        assert main_table[f"target_{metric}_ci_lower"].notna().all()
        assert main_table[f"target_{metric}_ci_upper"].notna().all()
    for metric in ("calibration_slope", "calibration_intercept"):
        assert f"target_{metric}_ci_lower" in main_table
        assert f"target_{metric}_ci_upper" in main_table
        eligible = main_table["frac_pred_at_clip_bounds"] <= 0.05
        assert main_table.loc[eligible, f"target_{metric}_ci_lower"].notna().all()
        assert main_table.loc[eligible, f"target_{metric}_ci_upper"].notna().all()
        assert main_table.loc[~eligible, f"target_{metric}_ci_lower"].isna().all()
        assert main_table.loc[~eligible, f"target_{metric}_ci_upper"].isna().all()
    assert len(bootstrap_calls) == 2 * len(main_table)
    assert "target_coverage_ci_lower" in first["conformal"]
    assert {"repeat", "median", "q1", "q3"}.issubset(
        set(first["repeats"]["row_type"])
    )


def test_repeat_slope_summaries_require_15_valid_repeats():
    rows = []
    for repeat in range(20):
        rows.append(
            {
                "split_set": "20_repeats",
                "repeat": repeat,
                "method": "A0",
                "model": "logistic",
                "calibrator": "none",
                "outcome": "NPM1",
                "analysis_role": "confirmatory",
                "status": "complete",
                "target_calibration_slope": (
                    float(repeat) if repeat < 14 else np.nan
                ),
                "target_calibration_intercept": (
                    float(repeat) if repeat < 15 else np.nan
                ),
                "source_test_calibration_slope": float(repeat),
                "source_test_calibration_intercept": (
                    float(repeat) if repeat < 14 else np.nan
                ),
            }
        )

    summaries = _append_repeat_summaries(rows)
    median = next(row for row in summaries if row["row_type"] == "median")

    assert median["target_calibration_slope_n_valid_repeats"] == 14
    assert np.isnan(median["target_calibration_slope"])
    assert median["target_calibration_intercept_n_valid_repeats"] == 15
    assert median["target_calibration_intercept"] == pytest.approx(7.0)
    assert median["source_test_calibration_slope_n_valid_repeats"] == 20
    assert median["source_test_calibration_intercept_n_valid_repeats"] == 14
    assert np.isnan(median["source_test_calibration_intercept"])


def test_target_plots_show_bootstrap_intervals_and_unconnected_counted_bins(
    tmp_path, monkeypatch
):
    from cohortshift import plots

    captured_figures = []
    original_subplots = plots.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured_figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plots.plt, "subplots", capture_subplots)
    reliability = pd.DataFrame(
        {
            "outcome": ["NPM1"] * 5,
            "method": ["A0"] * 5,
            "bin": [1, 2, 3, 4, 5],
            "bin_count": [35, 35, 35, 34, 34],
            "mean_predicted_probability": [0.1, 0.3, 0.5, 0.7, 0.9],
            "observed_fraction": [0.08, 0.28, 0.52, 0.68, 0.92],
        }
    )
    save_target_reliability_plot(
        reliability, tmp_path / "reliability_target.png"
    )
    reliability_figure = captured_figures[-1]
    reliability_axis = reliability_figure.axes[0]
    assert len(reliability_axis.lines) == 1
    assert len(reliability_axis.collections) == 1
    assert [text.get_text() for text in reliability_axis.texts] == [
        "35",
        "35",
        "35",
        "34",
        "34",
    ]

    coverage = pd.DataFrame(
        {
            "split_set": ["registered_main_split"],
            "model": ["logistic"],
            "calibrator": ["none"],
            "outcome": ["NPM1"],
            "status": ["complete"],
            "method": ["A0"],
            "target_coverage": [0.9],
            "target_coverage_ci_lower": [0.8],
            "target_coverage_ci_upper": [0.95],
            "target_positive_class_coverage": [0.85],
            "target_positive_class_coverage_ci_lower": [0.7],
            "target_positive_class_coverage_ci_upper": [0.95],
            "target_negative_class_coverage": [0.95],
            "target_negative_class_coverage_ci_lower": [0.85],
            "target_negative_class_coverage_ci_upper": [1.0],
        }
    )
    save_coverage_plot(coverage, tmp_path / "coverage.png")
    coverage_figure = captured_figures[-1]
    assert len(coverage_figure.axes) == 2
    assert all(
        any(isinstance(container, ErrorbarContainer) for container in axis.containers)
        for axis in coverage_figure.axes
    )

    for figure in captured_figures:
        plt.close(figure)


def test_target_bootstrap_metrics_are_bit_identical_for_same_seed():
    labels = np.tile([0, 1], 30)
    probabilities = np.where(labels == 1, 0.75, 0.25)
    patients = [f"patient-{index}" for index in range(len(labels))]
    first = _bootstrap_metric_intervals(
        labels, probabilities, patients, seed=190, n_resamples=25
    )
    second = _bootstrap_metric_intervals(
        labels, probabilities, patients, seed=190, n_resamples=25
    )
    assert first == second


def _e5_toy(n=120, seed=7):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(expit(rng.normal(-1 + 2 * y, 1.0)), 0.01, 0.99)
    p = p + np.arange(n) * 1e-9  # unique per-sample values
    ids = [f"S{i:03d}" for i in range(n)]
    return ids, p, y


def test_e5_never_evaluates_on_drawn_samples(monkeypatch):
    from cohortshift import recalibration as rc

    ids, p, y = _e5_toy()
    seen = []
    original = rc._prediction_metrics

    def spy(labels, probabilities):
        seen.append((len(labels), np.asarray(probabilities).copy()))
        return original(labels, probabilities)

    monkeypatch.setattr(rc, "_prediction_metrics", spy)
    draws, _ = rc.run_few_label_recalibration(
        ids, p, y, setting="t", seed=1, budgets=(10, 20), n_draws=3
    )
    assert (draws["n_eval"] == len(ids) - draws["k"]).all()
    assert len(seen) == len(draws)
    index = {s: i for i, s in enumerate(ids)}
    for row in draws.itertuples():
        drawn = row.drawn_samples.split(";")
        assert len(drawn) == row.k and len(set(drawn)) == row.k
        assert all(s in index for s in drawn)


def test_e5_drawn_samples_excluded_from_eval_predictions(monkeypatch):
    from cohortshift import recalibration as rc

    ids, p, y = _e5_toy()
    captured = []
    monkeypatch.setattr(
        rc,
        "_recalibrate",
        lambda method, dp, dy, new_p: (captured.append(new_p.copy()), new_p)[1],
    )
    draws, _ = rc.run_few_label_recalibration(
        ids, p, y, setting="t", seed=2, budgets=(10,), n_draws=2
    )
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    by_id = dict(zip(ids, pc))
    recalibrated_rows = draws.loc[draws["method"] != "none"]
    for row, new_p in zip(recalibrated_rows.itertuples(), captured):
        drawn_p = {by_id[s] for s in row.drawn_samples.split(";")}
        assert not drawn_p & set(new_p.tolist())


def test_e5_k0_equals_e3_metrics():
    from cohortshift import recalibration as rc
    from cohortshift.experiments import _prediction_metrics
    from cohortshift.source_experiments import _apply_calibrator

    ids, p, y = _e5_toy()
    _, summary = rc.run_few_label_recalibration(
        ids, p, y, setting="t", seed=3, budgets=(0,), n_draws=2
    )
    row = summary.iloc[0]
    expected = _prediction_metrics(y, _apply_calibrator("none", None, p))
    assert row["method"] == "none" and row["median_n_eval"] == len(y)
    assert row["brier_median"] == pytest.approx(expected["brier"])
    assert row["calibration_slope_median"] == pytest.approx(
        expected["calibration_slope"]
    )


def test_e5_draws_are_stratified_and_deterministic():
    from cohortshift.recalibration import draw_stratified_indices

    _, _, y = _e5_toy()
    first = draw_stratified_indices(y, 10, np.random.default_rng(5))
    second = draw_stratified_indices(y, 10, np.random.default_rng(5))
    assert (first == second).all() and len(first) == 10
    assert set(y[first]) == {0, 1}


def test_e5_intercept_only_keeps_slope_one():
    from cohortshift.recalibration import _fit_intercept_only
    from cohortshift.source_experiments import _clipped_logit

    ids, p, y = _e5_toy(n=200)
    shift = _fit_intercept_only(p, y)
    fitted = expit(_clipped_logit(p) + shift)
    assert fitted.sum() == pytest.approx(y.sum(), rel=1e-4)


def test_e5_summary_reports_failure_fraction():
    from cohortshift.recalibration import run_few_label_recalibration

    ids, p, y = _e5_toy()
    _, summary = run_few_label_recalibration(
        ids, p, y, setting="t", seed=4, budgets=(10,), n_draws=3
    )
    assert "fraction_fit_failed" in summary and len(summary) == 3


def test_e5_paired_differences_use_same_held_out_samples():
    from cohortshift.recalibration import _evaluate, run_few_label_recalibration

    ids, p, y = _e5_toy()
    draws, summary = run_few_label_recalibration(
        ids, p, y, setting="t", seed=9, budgets=(0, 10), n_draws=4
    )
    index = {sample: i for i, sample in enumerate(ids)}
    for draw in range(4):
        group = draws.loc[(draws["k"] == 10) & (draws["draw"] == draw)]
        assert set(group["method"]) == {"none", "intercept_only", "Platt"}
        assert group["drawn_samples"].nunique() == 1
        excluded = {index[s] for s in group.iloc[0]["drawn_samples"].split(";")}
        held = [i for i in range(len(ids)) if i not in excluded]
        baseline = group.set_index("method").loc["none"]
        expected = _evaluate(y[held], p[held])
        for metric in ("brier", "calibration_intercept", "calibration_slope"):
            assert baseline[metric] == pytest.approx(expected[metric])
            for _, row in group.loc[group["method"] != "none"].iterrows():
                if not row["fit_failed"]:
                    assert row[f"{metric}_difference_vs_none"] == pytest.approx(
                        row[metric] - baseline[metric]
                    )
    for method in ("intercept_only", "Platt"):
        group = draws.loc[(draws["k"] == 10) & (draws["method"] == method)]
        row = summary.loc[(summary["k"] == 10) & (summary["method"] == method)].iloc[0]
        for metric in ("brier", "calibration_intercept", "calibration_slope"):
            differences = group[f"{metric}_difference_vs_none"].dropna()
            for stat, q in (("median", 0.5), ("q1", 0.25), ("q3", 0.75)):
                assert row[f"{metric}_difference_vs_none_{stat}"] == pytest.approx(
                    differences.quantile(q)
                )
    assert pd.isna(
        summary.loc[summary["k"] == 0, "brier_difference_vs_none_median"].iloc[0]
    )


def test_e2_source_reference_is_separate_and_obeys_validity_rule():
    from cohortshift.experiments import add_e2_source_reference

    source = pd.DataFrame({
        "row_type": ["repeat"] * 20,
        "repeat": list(range(1, 21)),
        "seed": list(range(7101, 7121)),
        "model": ["logistic"] * 20,
        "calibrator": ["none"] * 20,
        "frac_pred_at_clip_bounds": [0.0] * 15 + [0.1] * 5,
        "brier": [0.1] * 20,
        "calibration_slope": list(range(20)),
        "calibration_intercept": list(range(20, 40)),
    })
    cross = pd.DataFrame({
        "model": ["logistic", "logistic"], "calibrator": ["none", "none"],
        "outcome": ["NPM1", "DNMT3A"], "source_repeat_median_calibration_slope": [99., 98.],
    })
    splits = pd.DataFrame({
        "repeat": list(range(1, 21)), "seed": list(range(7101, 7121)),
        "split_set": ["20_repeats"] * 20,
    })
    result = add_e2_source_reference(cross, source, splits)
    assert result.loc[0, "source_repeat_median_calibration_slope"] == 99.
    assert result.loc[0, "source_e2_median_calibration_slope"] == 7.
    assert result.loc[0, "source_e2_median_brier"] == pytest.approx(0.1)
    assert result.loc[0, "source_e2_calibration_slope_n_valid_repeats"] == 15
    assert pd.isna(result.loc[1, "source_e2_median_calibration_slope"])
    source.loc[14, "frac_pred_at_clip_bounds"] = 0.1
    assert pd.isna(add_e2_source_reference(cross, source, splits).loc[0, "source_e2_median_calibration_slope"])
    splits.loc[0, "seed"] = 1
    with pytest.raises(ValueError, match="seeds do not match"):
        add_e2_source_reference(cross, source, splits)


def test_e5_paired_calibration_differences_respect_clip_rule(monkeypatch):
    from cohortshift import recalibration as rc

    ids, p, y = _e5_toy()
    p[:30] = 1e-6
    monkeypatch.setattr(
        rc, "_recalibrate",
        lambda method, drawn_p, drawn_y, new_p: np.linspace(0.1, 0.9, len(new_p)),
    )
    draws, summary = rc.run_few_label_recalibration(
        ids, p, y, setting="t", seed=5, budgets=(10,), n_draws=2
    )
    methods = draws.loc[draws["method"] != "none"]
    assert methods["calibration_slope_difference_vs_none"].isna().all()
    assert methods["calibration_intercept_difference_vs_none"].isna().all()
    assert methods["brier_difference_vs_none"].notna().all()
    assert summary.loc[summary["method"] != "none", "calibration_slope_difference_vs_none_n_valid"].eq(0).all()


def test_repeated_split_table_has_no_patient_overlap_and_is_deterministic():
    from cohortshift.source_experiments import repeated_split_table

    samples = [f"P{i // 2:03d}_BA{i}" for i in range(80)]
    labels = pd.Series([int(i % 4 == 0) for i in range(80)], index=samples)
    patients = {s: s.split("_")[0] for s in samples}
    first = repeated_split_table(labels, patients, seeds=[1, 2])
    second = repeated_split_table(labels, patients, seeds=[1, 2])
    pd.testing.assert_frame_equal(first, second)
    for _, group in first.groupby("repeat"):
        assert set(group["sample_id"]) == set(samples)
        assert group.groupby("patient_id")["split"].nunique().max() == 1
