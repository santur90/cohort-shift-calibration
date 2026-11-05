# Cohort Shift Calibration

Reproducible experiments for mutation-label prediction and calibration under
cohort shift in acute myeloid leukemia (AML). The package supports Python 3.10
and newer.

## Pre-registration

This section is fixed before examining any target-cohort results.

- **Primary outcome:** NPM1 mutation status. Primary evaluation on TCGA-LAML
  uses Brier score, calibration intercept and slope, and AUROC.
- **Secondary outcome:** DNMT3A mutation status, evaluated only if the
  target analysis set (TCGA-LAML samples with RNA-seq) has at least 25
  profiled positives.
- **Mutation-label definition (both cohorts):** a sample is positive for a
  gene when `data_mutations.txt` has that gene and
  `Variant_Classification` is one of `Missense_Mutation`,
  `Nonsense_Mutation`, `Frame_Shift_Del`, `Frame_Shift_Ins`, `In_Frame_Del`,
  `In_Frame_Ins`, `Splice_Site`, `Translation_Start_Site`, or
  `Nonstop_Mutation`. A sample without the gene's mutation profiling is
  unlabelled, not negative. The Beat AML clinical `NPM1` column is not used as
  an outcome label.
- **Beat AML analysis unit:** retain samples with both RNA-seq and mutation
  profiling, then select one sample per patient. Prefer
  `DISEASE_STAGE_AT_SPECIMEN_COLLECTION == "Initial Diagnosis"`; print its
  distinct values and cross-check candidate records against `IS_RELAPSE` and
  `SPECIMEN_GROUPS`. Break ties by earliest
  `TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION`, then lowest sample ID.
  Patient ID is the sample ID without trailing `_BAxxxx` in Beat AML or
  trailing `-03` in TCGA-LAML.
- **Methods:** none (source-trained baseline); A1, per-cohort gene
  standardization; A2, within-sample gene-expression rank; and A3,
  domain-classifier reweighting with clipped weights.
- **No tuning on target results.** Target labels are reserved for evaluation;
  they are not used to select features, fit models, tune hyperparameters, or
  choose adaptation settings.
- **Adaptation functions never receive target labels.** Adaptation interfaces
  take source features and labels plus target features only where needed.

## Repository layout

- `src/cohortshift/`: data, label, feature, model, calibration, conformal,
  shift, adaptation, experiment, and plotting code.
- `scripts/fetch_data.sh`: selectively fetches the public cBioPortal datahub
  files listed below into `data/raw/<study_id>/`.
- `scripts/00_inventory.py`: inspects local files and writes
  `results/tables/inventory.tsv` and the pre-model analysis counts in
  `results/tables/analysis_set.tsv`.
- `config.tsv`: cohort IDs and RNA expression filenames.
- `results/tables/` and `results/figures/`: generated outputs.

## Data

The fetch script uses the public
[cBioPortal datahub](https://github.com/cBioPortal/datahub) Git LFS objects for
`aml_ohsu_2022` (Beat AML) and `laml_tcga_pan_can_atlas_2018` (TCGA-LAML).
It fetches clinical sample and patient tables, RNA expression, mutations, and
the gene-panel matrix, plus `data_sv.txt` (the structural-variant file present
in both study directories). The TCGA-LAML expression file is
`data_mrna_seq_v2_rsem.txt`; Beat AML uses `data_mrna_seq_rpkm.txt`.

Install Git LFS before running `make data`. The fetch script downloads only
these public study files and never requests GDC controlled-access reads.
Downloaded files live under `data/`, which is ignored by Git. **Do not add real
cohort data to version control.**

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
make data
make check-data
make run
make test
```

`make check-data` is a lightweight pre-analysis check: it prints the requested
clinical distinct values, candidate eligibility counts, and Beat AML
diagnosis/relapse cross-check without calculating mutation outcomes.
`make run` fails clearly if the expected input files have not been fetched and
writes the inventory and analysis-set counts.
The inventory reports each file's parsed header and inferred identifier
pattern, candidate clinical diagnosis/relapse columns, gene-panel coverage,
and per-cohort sample/patient/RNA and mutation-positive counts. Mutation
positives are counted only among samples marked profiled for that gene in the
panel matrix; unprofiled samples are excluded, not treated as negatives.
The cBioPortal panel matrix is sample-oriented (`SAMPLE_ID` and assay panel
columns). WXS/WGS profiles are treated as broad coverage of NPM1 and DNMT3A;
unknown targeted-panel IDs are not assumed to cover either gene.
For Beat AML, `analysis_set.tsv` records both the RNA-plus-mutation-profiled
candidate count and the one-sample-per-patient analysis count. DNMT3A's
registered 25-positive gate is calculated on TCGA-LAML RNA samples with known
mutation coverage, not from the cohort-wide mutation total.
