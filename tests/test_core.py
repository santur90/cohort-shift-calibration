import inspect

import numpy as np
import pandas as pd
import pytest

from cohortshift.adapt import adapt_features, adaptation_weights
from cohortshift.conformal import conformal_quantile, prediction_sets
from cohortshift.data import (
    find_header,
    iter_tsv,
    matrix_sample_columns,
    mutation_profile_samples,
    sample_id_format,
)
from cohortshift.features import cohort_standardize, within_sample_rank
from cohortshift.labels import (
    NON_SILENT_VARIANT_CLASSIFICATIONS,
    diagnosis_relapse_candidates,
    is_non_silent_variant,
    mutation_labels,
)
from cohortshift.shift import domain_classifier_weights


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


def test_conformal_quantile_and_prediction_sets():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.25) == 0.3
    assert prediction_sets(np.array([[0.9, 0.1]]), 0.2) == [(0,)]
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]))
