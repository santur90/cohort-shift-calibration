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
  input and fits an unpenalized logistic model with Platt-smoothed targets
  (`t+ = (N+ + 1)/(N+ + 2)`, `t- = 1/(N- + 2)`) using SciPy `minimize` with
  `method="L-BFGS-B"` and `maxiter=2000`. Summarize calibration slope and
  intercept by median, Q1, and Q3 rather than mean; summarize other metrics by
  mean and SD. Count repeats with slope above 5 and slope-fit non-convergence.
  Report paired per-repeat calibrator-minus-none differences for Brier and ECE,
  plus their mean and SD. Save five-bin reliability data and a source-only
  reliability plot.
- **E2 reporting decisions:** based on source-only results and before any
  target model result, the source reference for E3 is the median and IQR over
  the 20 repeats, with the registered main split shown separately. The main
  split was favorable (logistic AUROC 0.995 versus the 20-repeat median 0.968),
  so it is not used alone. Calibration slope and intercept are summarized by
  median and IQR, not mean. For Platt and isotonic, slope and intercept are
  reported as NA when more than 5% of predictions fall at either probability
  clip bound.
- **E2 logistic-grid correction:** the code's registered grid is
  `C in {0.001, 0.01, 0.1, 1}`. Correcting an earlier README statement,
  chosen C was 1.0 in 10/20 repeats (upper edge), 0.1 in 7/20, and 0.01 in
  3/20; the lower edge was never selected. The registered grid is kept.
- **E2 Platt and clip-bound decisions:** Platt uses Platt's smoothed targets,
  `t+ = (N+ + 1)/(N+ + 2)` and `t- = 1/(N- + 2)`. For any calibrator or
  setting, report calibration slope and intercept as NA when more than 5% of
  predictions are at either probability clip bound.
- **E3/E4 model selection:** the primary model is L2 logistic regression and
  the primary calibrator is none. HGB, Platt, and isotonic are secondary
  exploratory analyses. This choice is based on source-only E1/E2 results:
  Platt worsened Brier versus none in 19/20 repeats; isotonic was mixed, with
  a median of 81% of predictions at probability clip bounds. The choice was
  made after viewing source results and before any target model result.
- **E3/E4 expression scale and adaptations:** A0
  (`target log2(max(x, 0) + 1)`) is the primary no-adaptation setting; raw
  units are secondary. The registered adaptations are A0, A0+A1, A2, and
  A0+A3; raw A3 is also reported as a secondary adaptation. A2 is applied to
  raw target values and both cohorts because A0's `max(x, 0)` operation
  creates ties and is not strictly monotone. A3 uses the registered
  domain-classifier and clipping unchanged, normalizes weights to mean 1 for
  model training, and reports ESS, maximum weight, clip fractions, and the
  degenerate flag. A3 was degenerate in E0 (all weights at the 0.1 lower
  clip); label it as failed and do not tune it.
- **E3/E4 source reference and repeats:** compare with the 20-repeat source
  median and IQR and report the registered main split separately. The main
  split is favorable (logistic AUROC 0.995 versus 20-repeat median 0.968);
  never use the main split alone. For each of the 20 repeats, fit that
  repeat's own model using its own top-2,000 genes and apply it to all 173
  TCGA-LAML samples; report repeat results as median/IQR. Do not combine the
  20-repeat source of variability with the registered-main-split target
  bootstrap. Summarize calibration slope/intercept by median/IQR, not mean,
  and compare target slope/intercept with the source 20-repeat distribution,
  not with 1.
- **E3/E4 target uncertainty:** for the primary model trained on the
  registered 256/85/85 main split, estimate target uncertainty using 2,000
  patient-level bootstrap resamples and 95% percentile confidence intervals.
  Report the fraction of bootstrap resamples where the calibration-slope fit
  failed or the fitted slope exceeded 5.
- **Confirmatory comparisons:** the only confirmatory comparisons are
  (1) A0 logistic with no calibration target Brier score, calibration
  intercept, and calibration slope versus the 20-repeat source distribution;
  and (2) the paired patient-bootstrap difference in target Brier score,
  A0+A1 minus A0. All other E3/E4 outcomes and settings are exploratory.
- **E3/E4 lock:** run E3/E4 once. No method changes are allowed after viewing
  target results. Bug fixes are allowed only after disclosure in this README.
- **E3/E4 decision lock:** no method changes after viewing target results.
  Bug fixes are permitted only when disclosed in this README before changing
  the analysis.
- **Implementation bug-fix disclosure (before any E3/E4 real-data run):** the
  smoothed-target Platt optimizer's objective sign was corrected after a
  synthetic convergence test, and conformal prediction sets were extended to
  accept one-dimensional positive-class probabilities as well as two-column
  class probabilities. These fixes preserve the registered methods.
- **Additional pre-result implementation corrections:** before any E3/E4
  real-data run, code inspection found that the runner omitted raw A3, reused
  registered-main-split shift diagnostics for repeated splits, and did not
  report the registered paired A0+A1-minus-A0 target Brier bootstrap or the
  bootstrap fraction with failed/slope-above-5 calibration fits. The
  implementation will include raw A3, calculate shift diagnostics using each
  split's own selected genes, reuse the source E1/E2 training helper, and
  report those registered bootstrap quantities. These are implementation
  corrections; no target results were consulted.
- **E3/E4 reporting fixes (after viewing E3/E4 output):** populate the
  registered 2,000-resample target bootstrap confidence intervals for every
  complete main-split result row, not only the primary A0 row. Repeat
  slope/intercept summaries report the number of valid repeats and show NA
  summaries when fewer than 15 of 20 repeats have valid estimates under the
  existing clip-bound rule. Coverage plots show 95% bootstrap confidence
  intervals and positive- and negative-class coverage; reliability plots show
  points with bin-count annotations rather than connecting bins. These are
  reporting-only fixes; no method or setting changes.
- **Additional reporting corrections (after viewing E3/E4 output):** E5 now
  also evaluates the uncalibrated predictions on each draw's identical
  held-out samples and reports paired method-minus-none median and IQR for
  Brier score, intercept, and slope. The k=0 row is unchanged. For the main
  NPM1 analysis, E3/E4 tables separately report the saved E2 source-repeat
  Brier/slope/intercept reference (`source_e2_*`), in addition to the existing
  setting-specific E3 source-repeat estimates (`source_repeat_*`). E2 and E3
  use the same saved patient splits and genes, but E3 re-tunes C separately
  for each setting: E2 uses repeat seeds 7101–7120 for its shuffled training
  CV, while E3 uses 4201 plus the repeat number. Thus they can select
  different C values even for A0, where source features are unchanged; in
  the saved outputs C differed in 11 of 20 repeats. Both apply the >5%
  clip-bound NA rule. E3's source-reference median/IQR is additionally NA
  when fewer than 15 repeats are valid. The saved E2 reference uses this
  same minimum-valid rule when reported in E3. These additions correct the
  reference reporting without changing or refitting E3/E4 models.
- **Split conformal:** use `alpha=0.1` and nonconformity
  `1 - p(true class)`. Compute the finite-sample corrected quantile from
  source CALIBRATION scores as rank
  `ceil((n + 1) * (1 - alpha))`, capped at `n`. Report coverage overall and
  by class, mean prediction-set size, and empty-set fraction on source TEST
  and target samples under every registered setting. Bootstrap target
  coverage with the same 2,000 patient-level resamples and 95% percentile
  intervals.
- **E5 (few-label target recalibration; pre-registered before any E5 run):**
  the model is trained and calibrated (no calibrator) on source data only,
  exactly as in E3. The primary setting is A0+A1 with logistic regression
  and no calibrator. A0 is secondary and is reported with the note that most
  of its predictions sit at the probability clip bounds, so recalibration is
  ill-posed there. Target label budgets are `k in {0, 10, 20, 40}`. For each
  `k > 0`, draw 200 random target subsets stratified by label (proportional
  allocation with at least one sample per class where possible), with fixed
  saved seeds. Methods: (i) recalibration-in-the-large (intercept only, slope
  fixed at 1, fitted by offset) and (ii) Platt scaling with Platt's smoothed
  targets on `logit(p)`. The k drawn samples are used only for recalibration
  and are EXCLUDED from evaluation; each draw is used once. On the remaining
  target samples report the median and IQR over draws of Brier score,
  calibration intercept, and calibration slope, plus the fraction of draws
  where the fit failed (non-convergence, one-class draw, or slope fit
  failure). `k = 0` applies no recalibration and must equal the E3 result on
  all target samples. The E5 draws never feed back into any model, feature,
  or hyperparameter choice. E5 is exploratory.
- **E6a (sensitivity analysis):** restrict both cohorts to
  `ONCOTREE_CODE == "AML"` and exclude APL samples in both cohorts, using the
  registered clinical (`CANCER_TYPE_DETAILED`) plus structural-variant
  PML-RARA evidence union. Rerun E3/E4 in full, including regeneration of the
  60/20/20 source split (stratified on NPM1), the 20 repeated splits, and the
  top-2,000 gene selection on the new source TRAIN split. Print N and NPM1
  positives per cohort and per split before any model is fit. Methods and
  settings are unchanged from E3/E4.
- **E6b (exploratory):** use the earlier 585-sample source set (the
  fallback-allowed rule); rerun E3/E4 with the same unchanged methods,
  including split and gene-selection regeneration.
- E5, E6a, and E6b add no confirmatory comparisons: the only confirmatory
  comparisons remain the two registered above, and everything else is
  exploratory.
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

<!-- BEGIN GENERATED RESULTS -->
## Results

Generated from existing `results/tables` by `scripts/update_results.py`; NA means unavailable or suppressed by the registered clip rule. Only the two comparisons marked confirmatory are confirmatory; all other results below are exploratory.

### Main (exploratory)

- beat_aml: 426 samples, 121 NPM1 positives.
- tcga_laml: 173 samples, 48 NPM1 positives.

| Setting | Chosen hyperparameters / status | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE (95% CI) | Intercept (95% CI) | Slope (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | {"classifier__C": 1.0} | 0.977 [0.943, 1.000] | 0.966 [0.917, 0.999] | 0.275 [0.211, 0.343] | 0.276 [0.212, 0.345] | NA [NA, NA] | NA [NA, NA] |
| A0+A1 | {"C": 0.01} | 0.972 [0.926, 0.999] | 0.968 [0.924, 0.998] | 0.034 [0.014, 0.056] | 0.040 [0.020, 0.067] | -0.398 [-0.978, 0.254] | 1.066 [0.782, 2.865] |
| A2 | {"classifier__C": 0.1} | 0.976 [0.931, 0.999] | 0.972 [0.934, 0.998] | 0.034 [0.014, 0.057] | 0.036 [0.013, 0.065] | 1.735 [0.493, 3.187] | 0.703 [0.531, 1.666] |
| A0+A3 | failed | NA | NA | NA | NA | NA | NA |

Chosen hyperparameters by outcome, setting and model (main split):

| Outcome | Setting | Model | Status | Parameters |
| --- | --- | --- | --- | --- |
| NPM1 | raw | logistic | complete | {"classifier__C": 1.0} |
| NPM1 | raw | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 3} |
| NPM1 | A0 | logistic | complete | {"classifier__C": 1.0} |
| NPM1 | A0 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 3} |
| NPM1 | A0+A1 | logistic | complete | {"C": 0.01} |
| NPM1 | A0+A1 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 3} |
| NPM1 | A2 | logistic | complete | {"classifier__C": 0.1} |
| NPM1 | A2 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 3} |
| NPM1 | A3 | logistic | failed | NA |
| NPM1 | A3 | hist_gradient_boosting | failed | NA |
| NPM1 | A0+A3 | logistic | failed | NA |
| NPM1 | A0+A3 | hist_gradient_boosting | failed | NA |
| DNMT3A | raw | logistic | complete | {"classifier__C": 0.1} |
| DNMT3A | raw | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 3} |
| DNMT3A | A0 | logistic | complete | {"classifier__C": 0.1} |
| DNMT3A | A0 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 3} |
| DNMT3A | A0+A1 | logistic | complete | {"C": 1.0} |
| DNMT3A | A0+A1 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 3} |
| DNMT3A | A2 | logistic | complete | {"classifier__C": 0.1} |
| DNMT3A | A2 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 3} |
| DNMT3A | A3 | logistic | failed | NA |
| DNMT3A | A3 | hist_gradient_boosting | failed | NA |
| DNMT3A | A0+A3 | logistic | failed | NA |
| DNMT3A | A0+A3 | hist_gradient_boosting | failed | NA |

### E6a (exploratory)

- beat_aml: 406 samples, 121 NPM1 positives.
- tcga_laml: 157 samples, 48 NPM1 positives.

| Setting | Chosen hyperparameters / status | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE (95% CI) | Intercept (95% CI) | Slope (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | {"classifier__C": 0.01} | 0.973 [0.929, 0.998] | 0.969 [0.930, 0.996] | 0.183 [0.138, 0.233] | 0.227 [0.173, 0.285] | 3.803 [3.258, 4.431] | 1.127 [0.810, 2.488] |
| A0+A1 | {"C": 0.01} | 0.973 [0.930, 0.998] | 0.967 [0.926, 0.996] | 0.038 [0.018, 0.061] | 0.020 [0.012, 0.061] | -0.125 [-0.737, 0.573] | 1.036 [0.749, 2.262] |
| A2 | {"classifier__C": 1.0} | 0.975 [0.933, 0.998] | 0.970 [0.930, 0.996] | 0.041 [0.018, 0.067] | 0.038 [0.012, 0.072] | NA [NA, NA] | NA [NA, NA] |
| A0+A3 | failed | NA | NA | NA | NA | NA | NA |

Chosen hyperparameters by outcome, setting and model (main split):

| Outcome | Setting | Model | Status | Parameters |
| --- | --- | --- | --- | --- |
| NPM1 | raw | logistic | complete | {"classifier__C": 0.01} |
| NPM1 | raw | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| NPM1 | A0 | logistic | complete | {"classifier__C": 0.01} |
| NPM1 | A0 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| NPM1 | A0+A1 | logistic | complete | {"C": 0.01} |
| NPM1 | A0+A1 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| NPM1 | A2 | logistic | complete | {"classifier__C": 1.0} |
| NPM1 | A2 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| NPM1 | A3 | logistic | failed | NA |
| NPM1 | A3 | hist_gradient_boosting | failed | NA |
| NPM1 | A0+A3 | logistic | failed | NA |
| NPM1 | A0+A3 | hist_gradient_boosting | failed | NA |
| DNMT3A | raw | logistic | complete | {"classifier__C": 0.001} |
| DNMT3A | raw | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A0 | logistic | complete | {"classifier__C": 0.001} |
| DNMT3A | A0 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A0+A1 | logistic | complete | {"C": 0.001} |
| DNMT3A | A0+A1 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A2 | logistic | complete | {"classifier__C": 0.001} |
| DNMT3A | A2 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| DNMT3A | A3 | logistic | failed | NA |
| DNMT3A | A3 | hist_gradient_boosting | failed | NA |
| DNMT3A | A0+A3 | logistic | failed | NA |
| DNMT3A | A0+A3 | hist_gradient_boosting | failed | NA |

### E6b (exploratory)

- beat_aml: 585 samples, 152 NPM1 positives.
- tcga_laml: 173 samples, 48 NPM1 positives.

| Setting | Chosen hyperparameters / status | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE (95% CI) | Intercept (95% CI) | Slope (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | {"classifier__C": 0.001} | 0.972 [0.935, 0.997] | 0.959 [0.912, 0.993] | 0.200 [0.152, 0.251] | 0.230 [0.175, 0.290] | 2.984 [2.652, 3.294] | 1.947 [1.440, 4.215] |
| A0+A1 | {"C": 0.001} | 0.973 [0.936, 0.997] | 0.958 [0.911, 0.994] | 0.048 [0.031, 0.068] | 0.053 [0.038, 0.095] | 0.045 [-0.296, 0.374] | 1.808 [1.341, 3.926] |
| A2 | {"classifier__C": 0.001} | 0.975 [0.946, 0.997] | 0.952 [0.900, 0.992] | 0.077 [0.054, 0.103] | 0.111 [0.077, 0.148] | 1.076 [0.718, 1.413] | 1.778 [1.345, 3.602] |
| A0+A3 | failed | NA | NA | NA | NA | NA | NA |

Chosen hyperparameters by outcome, setting and model (main split):

| Outcome | Setting | Model | Status | Parameters |
| --- | --- | --- | --- | --- |
| NPM1 | raw | logistic | complete | {"classifier__C": 0.001} |
| NPM1 | raw | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| NPM1 | A0 | logistic | complete | {"classifier__C": 0.001} |
| NPM1 | A0 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| NPM1 | A0+A1 | logistic | complete | {"C": 0.001} |
| NPM1 | A0+A1 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 2} |
| NPM1 | A2 | logistic | complete | {"classifier__C": 0.001} |
| NPM1 | A2 | hist_gradient_boosting | complete | {"learning_rate": 0.05, "max_depth": 3} |
| NPM1 | A3 | logistic | failed | NA |
| NPM1 | A3 | hist_gradient_boosting | failed | NA |
| NPM1 | A0+A3 | logistic | failed | NA |
| NPM1 | A0+A3 | hist_gradient_boosting | failed | NA |
| DNMT3A | raw | logistic | complete | {"classifier__C": 0.001} |
| DNMT3A | raw | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A0 | logistic | complete | {"classifier__C": 0.001} |
| DNMT3A | A0 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A0+A1 | logistic | complete | {"C": 0.001} |
| DNMT3A | A0+A1 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A2 | logistic | complete | {"classifier__C": 0.001} |
| DNMT3A | A2 | hist_gradient_boosting | complete | {"learning_rate": 0.1, "max_depth": 2} |
| DNMT3A | A3 | logistic | failed | NA |
| DNMT3A | A3 | hist_gradient_boosting | failed | NA |
| DNMT3A | A0+A3 | logistic | failed | NA |
| DNMT3A | A0+A3 | hist_gradient_boosting | failed | NA |

### Confirmatory comparisons

1. A0 logistic/no-calibration target versus the saved E2 source reference:

| Metric | Target (95% CI) | E2 source median [Q1, Q3] | E2 source main split |
| --- | --- | --- | --- |
| brier | 0.275 [0.211, 0.343] | 0.026 [0.014, 0.031] | 0.021 |
| calibration_intercept | NA [NA, NA] | 0.327 [0.023, 0.901] | -0.507 |
| calibration_slope | NA [NA, NA] | 0.771 [0.635, 1.015] | 0.915 |

2. Paired target Brier difference, A0+A1 minus A0: -0.241 [-0.309, -0.176].

The E2 source reference uses the saved source-only training runs; the setting-specific E3 source repeat summaries are separate. C is re-tuned for every E3 setting. E2 uses CV seeds 7101–7120; E3 uses 4201 plus repeat number, with the same patient splits and genes. Both use the >5% clip-bound NA rule; the E3 summaries and this report suppress slope/intercept when fewer than 15 repeats are valid.

### E5 (exploratory)

| Setting | k | Method | Brier [Q1, Q3] | Paired Δ Brier | Intercept [Q1, Q3] | Paired Δ intercept | Slope [Q1, Q3] | Paired Δ slope | Fit failure fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0+A1 | 0 | none | 0.034 [0.034, 0.034] | NA [NA, NA] | -0.398 [-0.398, -0.398] | NA [NA, NA] | 1.066 [1.066, 1.066] | NA [NA, NA] | 0.000 |
| A0+A1 | 10 | Platt | 0.056 [0.053, 0.060] | 0.021 [0.018, 0.027] | -0.233 [-0.443, -0.039] | 0.167 [0.015, 0.327] | 2.419 [2.232, 2.619] | 1.378 [1.191, 1.564] | 0.000 |
| A0+A1 | 10 | intercept_only | 0.036 [0.033, 0.042] | 0.002 [-0.002, 0.008] | -0.215 [-0.915, 0.464] | 0.191 [-0.479, 0.837] | 1.044 [1.040, 1.051] | -0.000 [-0.000, 0.000] | 0.000 |
| A0+A1 | 10 | none | 0.035 [0.034, 0.035] | 0.000 [0.000, 0.000] | -0.407 [-0.446, -0.370] | 0.000 [0.000, 0.000] | 1.045 [1.040, 1.051] | 0.000 [0.000, 0.000] | 0.000 |
| A0+A1 | 20 | Platt | 0.045 [0.043, 0.048] | 0.009 [0.007, 0.018] | -0.181 [-0.405, 0.025] | 0.241 [0.101, 0.386] | 1.783 [1.661, 2.324] | 0.745 [0.639, 1.159] | 0.000 |
| A0+A1 | 20 | intercept_only | 0.035 [0.034, 0.040] | 0.000 [-0.002, 0.006] | -0.246 [-0.854, 0.408] | 0.177 [-0.368, 0.744] | 1.027 [1.018, 1.169] | 0.000 [-0.000, 0.000] | 0.000 |
| A0+A1 | 20 | none | 0.035 [0.031, 0.037] | 0.000 [0.000, 0.000] | -0.423 [-0.503, -0.354] | 0.000 [0.000, 0.000] | 1.027 [1.019, 1.170] | 0.000 [0.000, 0.000] | 0.000 |
| A0+A1 | 40 | Platt | 0.039 [0.037, 0.041] | 0.005 [0.001, 0.008] | -0.084 [-0.281, 0.168] | 0.356 [0.239, 0.493] | 1.452 [1.266, 2.010] | 0.422 [0.277, 0.790] | 0.000 |
| A0+A1 | 40 | intercept_only | 0.035 [0.030, 0.038] | -0.001 [-0.002, 0.003] | -0.077 [-0.632, 0.368] | 0.336 [-0.086, 0.675] | 1.032 [0.975, 1.215] | 0.000 [-0.000, 0.000] | 0.000 |
| A0+A1 | 40 | none | 0.034 [0.029, 0.039] | 0.000 [0.000, 0.000] | -0.414 [-0.520, -0.310] | 0.000 [0.000, 0.000] | 1.032 [0.975, 1.215] | 0.000 [0.000, 0.000] | 0.000 |
| A0 | 0 | none | 0.275 [0.275, 0.275] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.000 |
| A0 | 10 | Platt | 0.049 [0.044, 0.055] | -0.224 [-0.229, -0.219] | -0.171 [-0.381, -0.013] | NA [NA, NA] | 2.931 [2.664, 3.267] | NA [NA, NA] | 0.000 |
| A0 | 10 | intercept_only | 0.031 [0.028, 0.033] | -0.242 [-0.245, -0.241] | 0.246 [-0.639, 0.909] | NA [NA, NA] | 1.264 [1.247, 1.267] | NA [NA, NA] | 0.000 |
| A0 | 10 | none | 0.273 [0.273, 0.273] | 0.000 [0.000, 0.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.000 |
| A0 | 20 | Platt | 0.037 [0.034, 0.040] | -0.235 [-0.238, -0.232] | -0.148 [-0.408, 0.123] | NA [NA, NA] | 2.249 [2.024, 2.502] | NA [NA, NA] | 0.000 |
| A0 | 20 | intercept_only | 0.029 [0.026, 0.032] | -0.242 [-0.246, -0.239] | -0.052 [-0.872, 0.760] | NA [NA, NA] | 1.251 [1.230, 1.347] | NA [NA, NA] | 0.000 |
| A0 | 20 | none | 0.271 [0.271, 0.272] | 0.000 [0.000, 0.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.000 |
| A0 | 40 | Platt | 0.031 [0.028, 0.035] | -0.244 [-0.248, -0.240] | -0.043 [-0.272, 0.267] | NA [NA, NA] | 1.806 [1.565, 2.063] | NA [NA, NA] | 0.000 |
| A0 | 40 | intercept_only | 0.027 [0.024, 0.034] | -0.248 [-0.251, -0.241] | -0.096 [-0.440, 0.524] | NA [NA, NA] | 1.291 [1.199, 1.437] | NA [NA, NA] | 0.000 |
| A0 | 40 | none | 0.275 [0.275, 0.276] | 0.000 [0.000, 0.000] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | NA [NA, NA] | 0.000 |

The A0 recalibration-is-ill-posed warning was a pre-result hypothesis; observed recalibrated A0 predictions did not remain at the clip bounds. This exploratory observation does not establish generalizability.

### E7 / secondary-label DNMT3A analysis (exploratory)

E7 secondary-label cross-cohort results are stored in the E3/E4 `e3_e4_cross_cohort*.tsv` tables (`outcome == DNMT3A`), not in a separate E7 file. This descriptive summary is limited to the preregistered primary no-adaptation setting (A0, logistic, no calibrator); DNMT3A analyses are exploratory.

| Analysis set | Target DNMT3A positives/profiled | Status | AUROC (target 95% CI) | Brier (target 95% CI) | Intercept (target 95% CI) | Slope (target 95% CI) | Source-repeat Brier median [Q1, Q3] | Source-repeat intercept median [Q1, Q3] | Source-repeat slope median [Q1, Q3] | Source main-split Brier | Source main-split intercept | Source main-split slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main | 43/173 | complete | 0.826 [0.733, 0.899] | 0.209 [0.156, 0.265] | 4.363 [3.668, 5.069] | 0.488 [0.314, 0.716] | 0.115 [0.108, 0.122] | 0.080 [-0.073, 0.302] | 0.843 [0.451, 1.266] | 0.180 | 1.710 | 0.300 |
| E6a | 43/157 | complete | 0.787 [0.697, 0.867] | 0.190 [0.142, 0.241] | 1.188 [0.796, 1.551] | 1.364 [0.913, 1.975] | 0.116 [0.106, 0.152] | 0.765 [-0.043, 1.071] | 0.575 [0.456, 1.159] | 0.107 | -0.362 | 1.706 |
| E6b | 43/173 | complete | 0.816 [0.730, 0.886] | 0.184 [0.139, 0.233] | 1.505 [1.131, 1.846] | 1.419 [0.990, 1.973] | 0.110 [0.098, 0.118] | 0.306 [-0.059, 1.287] | 0.682 [0.376, 0.914] | 0.154 | 0.744 | 1.774 |

Source-repeat columns are the saved E3/E4 source-test summaries over 20 repeats. Source main-split metrics are shown separately. Target intervals are the existing patient-bootstrap intervals. This table summarizes results only and does not add a confirmatory comparison.

### Split conformal (exploratory)

| Setting | Source coverage | Target coverage (95% CI) | Target positive | Target negative | Mean set size | Empty fraction |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0.906 | 0.659 [0.589, 0.728] | 0.604 | 0.680 | 1.000 | 0.000 |
| A0 | 0.906 | 0.723 [0.653, 0.786] | 0.000 | 1.000 | 1.000 | 0.000 |
| A0+A1 | 0.918 | 0.902 [0.855, 0.948] | 0.917 | 0.896 | 0.925 | 0.075 |
| A2 | 0.894 | 0.884 [0.832, 0.931] | 0.646 | 0.976 | 0.902 | 0.098 |
<!-- END GENERATED RESULTS -->

## Limitations

The target cohort is small (173 samples and 48 NPM1 positives in the main
set; 157 and 48 in E6a), with wide confidence intervals; interpret results as
trends only. This is a methods demonstration on public data, not a clinical
model or a biological discovery. Beat AML expression is log2 RPKM and TCGA
expression is batch-corrected RSEM; A0 was chosen after viewing target value
ranges, not target labels. A3 failed and was not tuned. E0 was first run on
the earlier 585-sample source set. Source test splits are small (about 85
patients, 81 in E6a, and 116 in E6b); main-split slopes disagree with the
20-repeat slopes. Source-test split conformal coverage is already below the
nominal 0.9 for some settings at these sizes.

Beat AML data are CC-BY-4.0; no real data are committed. Cite Bottomly et al.,
["Integrative analysis of drug response and clinical outcome in acute myeloid leukemia"](https://doi.org/10.1016/j.ccell.2022.08.013),
*Cancer Cell* (2022), and The Cancer Genome Atlas Research Network,
["Genomic and epigenomic landscapes of adult de novo acute myeloid leukemia"](https://doi.org/10.1056/NEJMoa1301689),
*New England Journal of Medicine* (2013).

## Deviations and post-hoc decisions

Chronologically, with decision timing disclosed:

1. After E0 was run on the 585-sample source set and target expression (not
   target labels), the main source set was restricted to initial diagnosis
   based on source clinical composition; E6b retains the earlier rule.
2. After examining target expression availability, missing genes were excluded
   using availability only, without labels or model results. After viewing
   target value ranges and raw A3 degeneracy, A0 was selected; A2 uses raw
   values because A0 can create ties.
3. After source-only E1/E2 results, the source 20-repeat median/IQR was made
   the reference alongside the separately reported favorable main split;
   Platt smoothing and clip-bound exclusions were adopted. The grid was not
   changed. The initially misstated C-grid edge counts were corrected.
4. After source results and before target model results, logistic/no
   calibration was designated primary; HGB and calibration were exploratory.
5. After E3/E4 output, confidence-interval and plot reporting were corrected;
   valid-repeat counts were added. Subsequently, paired E5 reporting and the
   separate E2 source reference were added, without changing fitted methods.
6. After E5 output, the prior warning that A0 recalibration might be
   ill-posed did not hold for the observed recalibrated predictions; it
   remains a pre-result warning, not a conclusion.

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
- `scripts/04_cross_cohort.py`: runs the locked E3/E4 and split-conformal
  cross-cohort analyses once.
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
`make e3` runs the preregistered cross-cohort evaluation once. It uses saved
source-train gene lists, source-only training/calibration, and target features
for unsupervised adaptations; target labels are read only inside the final
evaluation function. Outputs are
`results/tables/e3_e4_cross_cohort.tsv`,
`results/tables/e3_e4_repeats.tsv`,
`results/tables/conformal_coverage.tsv`,
`results/figures/reliability_target.png`, and
`results/figures/coverage.png`.
`make e5` runs the pre-registered few-label recalibration
(`scripts/05_few_label.py`) on frozen source-trained E3 predictions and writes
`results/tables/e5_few_label.tsv` (median/IQR summaries),
`results/tables/e5_few_label_draws.tsv` (per-draw rows with drawn sample IDs
and paired differences),
and `results/splits/e5_seeds.json`. `make e6` regenerates the E6a (AML,
APL excluded) and E6b (earlier 585-sample source set) analysis sets with
`scripts/01_prepare.py --analysis-set` and reruns E3/E4 on each, writing
`_e6a`/`_e6b`-suffixed tables and figures.
`make run` fails clearly if the expected input files have not been fetched and
writes the inventory and analysis-set counts.
`make smoke` runs preparation, source experiments, cross-cohort evaluation,
few-label recalibration, conformal evaluation and plotting on generated,
shifted synthetic data in a temporary directory. The fixture description is
under `tests/fixtures/`; no real data are read. To keep this wiring check fast,
it uses one source repeat, 16 selected genes, 10 bootstrap resamples and two
E5 draws, rather than the registered real-data budgets. `make reproduce-check`
runs it twice and compares the SHA-256 hashes of every generated TSV table,
including compressed expression matrices. Gzip timestamps are fixed to make
the serialized outputs reproducible; their contents and methods are unchanged.
Run `python scripts/update_results.py` to replace the marked results section
with values read from existing `results/tables`; it does not run experiments.
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
