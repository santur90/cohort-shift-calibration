# Cohort Shift Calibration

Reproducible experiments for mutation-label prediction and calibration under
cohort shift in acute myeloid leukemia (AML). The package supports Python 3.10
and newer.

## Pre-registration

This section was finalized before examining target labels or model outputs.
The earlier expression-only E0 run and subsequent source-only analysis-set
revision are disclosed below.

- **Main analysis set (Beat AML):** retain samples with both RNA-seq and
  mutation profiling and
  `DISEASE_STAGE_AT_SPECIMEN_COLLECTION == "Initial Diagnosis"`, then select
  one sample per patient. Break ties by earliest
  `TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION`, then lowest sample ID.
  Patients without an Initial Diagnosis candidate are excluded; there is no
  fallback to another stage.
- **Rationale:** TCGA-LAML consists of untreated primary samples, so the source
  main set is restricted to initial-diagnosis samples to match clinical state.
  The decision was based only on the source-cohort stage composition (426 of
  585 selected samples were initial diagnosis; 159 patients had no
  initial-diagnosis sample).
- **Decision timing:** E0 was first run on the 585-sample set using target
  expression but no target labels and no model outputs; the analysis set was
  then revised using source-cohort composition only.
- **Composition:** non-AML samples remain in the main set, including 5 ALAL
  samples among the 426 initial-diagnosis samples.
- **APL definition (identical for both cohorts):** classify a sample as APL if
  its clinical diagnosis contains an acute promyelocytic leukaemia label or
  PML-RARA, or its structural-variant record has PML and RARA as the two
  partner genes or reports a PML-RARA fusion. The Beat AML source check found
  the SV columns `Sample_Id`, `NCBI_Build`, `Site1_Hugo_Symbol`,
  `Site1_Description`, `Site2_Hugo_Symbol`, `SV_Status`, and `Event_Info`;
  PML-RARA calls appear as `Site1_Hugo_Symbol = PML`,
  `Site2_Hugo_Symbol = RARA`, and `Event_Info = PML-RARA Fusion`. The clinical
  field is `CANCER_TYPE_DETAILED`, with observed APL value
  `Acute promyelocytic leukaemia with t(15;17)(q22;q12); PML-RARA`.
  The TCGA-LAML `data_sv.txt` schema is `Sample_Id`, `Site1_Hugo_Symbol`,
  `Site1_Chromosome`, `Site1_Position`, `Site2_Hugo_Symbol`,
  `Site2_Chromosome`, `Site2_Position`, `Site2_Effect_On_Frame`, `SV_Status`,
  `Tumor_Split_Read_Count`, `Tumor_Paired_End_Read_Count`, `NCBI_Build`,
  `Event_Info`, and `Center`. Public-file inspection found 23 Beat AML and 16
  TCGA-LAML PML-RARA SV records by both partner-gene and event-info evidence.
- **Expression gene availability:** Genes not measured (non-missing) in every
  sample of both cohorts are excluded before feature selection. This decision
  uses gene availability only, no labels or model results.
- **A0 target-expression transform:** for target-cohort expression values
  `x`, replace each value with `log2(max(x, 0) + 1)`; source expression is
  unchanged because Beat AML is already log2. A0 uses no statistics and no
  labels. It is the primary no-adaptation setting; raw units are reported as
  secondary analyses. A1 and A3 are applied on top of A0 and reported as
  `A0+A1` and `A0+A3`. A2 is invariant to monotone transforms and is reported
  once.
- **A0 decision disclosure:** A0 was chosen after viewing target expression
  value ranges, not target labels. A3 on raw units was already observed to be
  degenerate, with all weights at the 0.1 lower clip.
- **A3 diagnostics:** report effective sample size (ESS), maximum weight, and
  clip bounds, minimum and median weights, and the fractions at each clip
  bound. The degenerate-weight flag is true when ESS is below 10% of N, the
  maximum weight reaches the upper clip, at least half the weights reach either
  clip, or all weights are equal; the output records the reason. The domain
  AUROC remains unweighted and unchanged by construction.
- **A1 E0 domain AUROC:** fit per-cohort gene means and standard deviations
  within each cross-validation training fold and apply them to its held-out
  fold. Also report the prior whole-data-standardized AUROC separately; this
  standardization forces equal class means and biases CV AUROC below 0.5.
- **E1/E2 source-only protocol:** use Beat AML main-set expression, NPM1 labels,
  the saved 14,005 jointly retained gene names, and the saved list of 426
  Beat AML main-set patients only. Never load target-cohort expression or
  labels. Run 20 repeated patient-stratified 60/20/20 train/calibration/test
  splits with seeds 7101 through 7120. First reserve 40% as a holdout, then
  split it equally into calibration and test sets; stratify both steps on
  NPM1. For each repeat, select the 2,000 highest-variance genes using that
  repeat's training samples only. Tune only within training using 5-fold
  stratified CV maximizing AUROC. Also report the registered 256/85/85 main
  split as separate rows; it is not one of the 20 repeats.
- **E1 models and fixed grids:** L2 logistic regression in a
  `StandardScaler`/logistic pipeline, with `C` in `{0.001, 0.01, 0.1, 1}`;
  and `HistGradientBoostingClassifier`, with `max_depth` in `{2, 3}`,
  `learning_rate` in `{0.05, 0.1}`, `max_iter=200`, `early_stopping=True`,
  and `l2_regularization=1.0`. Report AUROC, AUPRC, Brier score, and ECE
  using 5 equal-count bins per test split, including bin counts. Summarize
  metric means and sample standard deviations across repeats. Each test split
  has about 85 patients and about 24 positives, so single-split metrics are
  noisy.
- **E2 calibration:** for each tuned model and repeat, fit none, Platt, and
  isotonic calibrators on that repeat's calibration split only (about 85
  patients); isotonic calibration is noisy at this size. Evaluate the same
  test metrics plus calibration slope and calibration intercept. Clip
  probabilities to `[1e-6, 1 - 1e-6]` before calculating logits. Estimate
  calibration slope from a logistic regression with free intercept and slope;
  estimate calibration intercept from a logistic model with slope fixed at 1
  using the logit as an offset. Platt takes `logit(p)` as its one-dimensional
  input, uses an unpenalized logistic fit (`C=np.inf`) with
  `solver="lbfgs"` and `max_iter=2000`. In addition to mean and SD, report median, Q1, and Q3
  for slope and intercept; count repeats with slope above 5 and slope-fit
  non-convergence. Report paired per-repeat calibrator-minus-none differences
  for Brier and ECE, plus their mean and SD. Save five-bin reliability data and
  a source-only reliability plot.
- **E2 reporting decisions:** based on source-only results and before any
  target model result, the source reference for E3 is the median and IQR over
  the 20 repeats, with the registered main split shown separately. The main
  split was favorable (logistic AUROC 0.995 versus the 20-repeat median 0.968),
  so it is not used alone. Calibration slope and intercept are summarized by
  median and IQR, not mean. For Platt and isotonic, slope and intercept are
  reported as NA when more than 5% of predictions fall at either probability
  clip bound.
- **E2 hyperparameter and Platt decisions:** the registered logistic C grid is
  retained. Chosen C was 1.0 in 10/20 repeats (upper edge) and 0.01 in 3/20
  repeats (lower edge), for 13/20 repeats at a grid edge. Platt was changed to
  use Platt's smoothed targets,
  `t+ = (N+ + 1)/(N+ + 2)` and `t- = 1/(N- + 2)`, because the prior
  unpenalized fit produced hard 0/1 outputs on near-separable calibration sets.
- **E6a (sensitivity analysis):** restrict to `ONCOTREE_CODE == "AML"` and
  exclude APL samples in both cohorts; rerun E3/E4.
- **E6b (exploratory):** add back the fallback samples selected by the
  previous 585-sample rule; rerun E3/E4.
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
- **Patient IDs:** the sample ID without trailing `_BAxxxx` in Beat AML or
  trailing `-03` in TCGA-LAML.
- **Methods:** raw-unit rows are retained for none (source-trained baseline),
  A1 (per-cohort gene standardization), A2 (within-sample gene-expression
  rank), and A3 (domain-classifier reweighting with clipped weights). Also
  report A0, A0+A1, and A0+A3; A2 is reported only once.
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
- `scripts/01_prepare.py`: builds aligned expression matrices under ignored
  `data/processed/`, final registered analysis-set labels, patient splits, and
  source-train-only top-variance gene selection.
- `scripts/02_e0_shift.py`: runs E0 domain-classifier AUROC, PCA overlap, and
  per-gene KS diagnostics for the requested methods.
- `scripts/03_source_experiments.py`: runs source-only E1/E2 with repeated
  patient splits and writes baseline, calibration, and reliability outputs.
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
make check-apl
make prepare
make e0
make e1
make e2
make run
make test
```

`make check-data` is a lightweight pre-analysis check: it prints the requested
clinical distinct values, candidate eligibility counts, and Beat AML
diagnosis/relapse cross-check without calculating mutation outcomes.
`make check-apl` is read-only and reports PML-RARA evidence counts from clinical
diagnoses and structural variants for each full cohort and its main analysis
set; it does not read mutation labels or write results.
`make prepare` writes gzipped aligned matrices to the ignored `data/processed/`
directory, `results/splits/` patient assignments and fixed seeds, and the final
`analysis_set.tsv`. Beat AML patient splits are deterministic 60/20/20
train/validation/test partitions stratified on the registered source label;
the TCGA-LAML split is not stratified using target labels. Cohorts with fewer
than three patients use only as many partitions as possible. The top
2,000 genes are ranked by variance on Beat AML's training split only, after
excluding genes with any missing values in either cohort. Preparation reports
genes absent from one cohort, shared genes incomplete in either cohort, and
the number retained; the main analysis set requires exactly 14,005 retained
genes. It also prints descriptive selected-gene expression summaries without
using them to make decisions. Main patient splits
are stratified by NPM1; a separate DNMT3A-stratified split is written for E7.
Split and E0 random seeds are recorded in `results/splits/seeds.json`.
Beat AML NPM1/DNMT3A positive counts by split are saved in
`results/splits/beat_aml_split_counts.tsv`; no target-label split counts are
written.
`make e0` writes method-level metrics to `results/tables/e0_shift.tsv`,
per-gene statistics to `results/tables/e0_gene_ks.tsv`, and one PCA overlap
figure per method. A1 domain AUROC uses fold-local per-cohort standardization;
the separate whole-data-standardized AUROC is retained for comparison. The
preparation step prints descriptive expression-value summaries for the
selected 2,000 genes without using them to make decisions. Beat AML uses log2
RPKM while TCGA-LAML uses batch-corrected
RSEM; this expression-unit difference is part of the shift and is reported as
such. Genes with any missing value in either cohort are excluded before
feature selection, so E0 domain-classifier and PCA diagnostics do not impute
expression values. E0's A3 summary reports the effective sample size, maximum
weight, minimum and median weights, clip bounds, fractions at each clip, and
degenerate-weight status with its reason.
`make e1` and `make e2` run source-only experiments without loading target
expression or labels. Preparation saves the jointly retained gene names to
`data/processed/jointly_complete_gene_names.txt`; E1/E2 require that list,
the Beat AML aligned expression and NPM1 labels, and the saved Beat AML
main-set patient list. E1 writes `results/tables/e1_baseline.tsv` and E2
writes `results/tables/e2_calibration.tsv`,
`results/tables/e2_reliability.tsv`, and
`results/figures/reliability_source.png`. Twenty-repeat patient split
assignments and seeds are saved under `results/splits/`.
The E2 table keeps all repeat-level rows, including selected hyperparameters,
and separate registered-main-split rows marked as not part of the 20 repeats.
The runner prints the selected logistic C and HGB parameter distributions and
flags when logistic C is at a grid edge in more than half of the 20 repeats.
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
