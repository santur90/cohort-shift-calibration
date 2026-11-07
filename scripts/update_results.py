#!/usr/bin/env python3
"""Update the marked README results section from existing result tables only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def number(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def interval(row: pd.Series, column: str) -> str:
    return (
        f"{number(row[column])} "
        f"[{number(row[column + '_ci_lower'])}, {number(row[column + '_ci_upper'])}]"
    )


def rows_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def read_table(tables: Path, name: str) -> pd.DataFrame | None:
    path = tables / name
    return pd.read_csv(path, sep="\t") if path.is_file() else None


def select(frame: pd.DataFrame, **filters: str) -> pd.DataFrame:
    for column, value in filters.items():
        frame = frame.loc[frame[column] == value]
    return frame


def report(tables: Path) -> str:
    lines = [
        "## Results",
        "",
        "Generated from existing `results/tables` by `scripts/update_results.py`; "
        "NA means unavailable or suppressed by the registered clip rule. "
        "Only the two comparisons marked confirmatory are confirmatory; all other "
        "results below are exploratory.",
    ]
    main = read_table(tables, "e3_e4_cross_cohort.tsv")
    e2 = read_table(tables, "e2_calibration.tsv")
    frames = {}
    for label, suffix in (("Main", ""), ("E6a", "_e6a"), ("E6b", "_e6b")):
        frame = read_table(tables, f"e3_e4_cross_cohort{suffix}.tsv")
        counts = read_table(tables, f"analysis_set{suffix}.tsv")
        if frame is None:
            lines += [f"\n### {label} (exploratory)", "\nResults unavailable."]
            continue
        all_models = select(frame, calibrator="none")
        frame = select(all_models, model="logistic", outcome="NPM1")
        frames[label] = frame
        lines += [f"\n### {label} (exploratory)", ""]
        if counts is not None:
            for _, row in counts.iterrows():
                lines.append(
                    f"- {row['cohort']}: {int(row['n_samples'])} samples, "
                    f"{int(row['npm1_positives'])} NPM1 positives."
                )
        table_rows = []
        for method in ("A0", "A0+A1", "A2", "A0+A3"):
            matches = select(frame, method=method)
            if len(matches) != 1:
                raise ValueError(f"{label} {method}: expected one primary row")
            row = matches.iloc[0]
            if row["status"] != "complete":
                table_rows.append([method, str(row["status"]), "NA", "NA", "NA", "NA", "NA", "NA"])
                continue
            table_rows.append([
                method, str(row["hyperparameters"]), interval(row, "target_auroc"),
                interval(row, "target_auprc"), interval(row, "target_brier"),
                interval(row, "target_ece"), interval(row, "target_calibration_intercept"),
                interval(row, "target_calibration_slope"),
            ])
        lines += ["", rows_table(
            ["Setting", "Chosen hyperparameters / status", "AUROC (95% CI)",
             "AUPRC (95% CI)", "Brier (95% CI)", "ECE (95% CI)",
             "Intercept (95% CI)", "Slope (95% CI)"], table_rows
        )]
        hyper_rows = [
            [str(row["outcome"]), str(row["method"]), str(row["model"]), str(row["status"]),
             "NA" if pd.isna(row["hyperparameters"]) else str(row["hyperparameters"])]
            for _, row in all_models.iterrows()
        ]
        lines += ["", "Chosen hyperparameters by outcome, setting and model (main split):", "",
                  rows_table(["Outcome", "Setting", "Model", "Status", "Parameters"], hyper_rows)]
    if main is not None and e2 is not None:
        a0 = select(frames["Main"], method="A0").iloc[0]
        source = select(e2, row_type="repeat", model="logistic", calibrator="none")
        main_source = select(e2, row_type="registered_main_split", model="logistic", calibrator="none")
        if len(source) != 20 or len(main_source) != 1:
            raise ValueError("Expected 20 E2 repeats and one registered main split")
        lines += ["\n### Confirmatory comparisons", "",
                  "1. A0 logistic/no-calibration target versus the saved E2 source reference:", ""]
        comparison = []
        for metric in ("brier", "calibration_intercept", "calibration_slope"):
            values = pd.to_numeric(source[metric], errors="raise").dropna()
            if metric != "brier" and len(values) < 15:
                reference = "NA"
            else:
                reference = f"{number(values.median())} [{number(values.quantile(.25))}, {number(values.quantile(.75))}]"
            comparison.append([metric, interval(a0, f"target_{metric}"), reference,
                               number(main_source.iloc[0][metric])])
        lines += [rows_table(["Metric", "Target (95% CI)", "E2 source median [Q1, Q3]",
                              "E2 source main split"], comparison), "",
                  "2. Paired target Brier difference, A0+A1 minus A0: "
                  + interval(a0, "target_brier_difference_a0_a1_minus_a0") + "."]
        lines += ["", "The E2 source reference uses the saved source-only training runs; "
                  "the setting-specific E3 source repeat summaries are separate. "
                  "C is re-tuned for every E3 setting. E2 uses CV seeds 7101–7120; "
                  "E3 uses 4201 plus repeat number, with the same patient splits and genes. "
                  "Both use the >5% clip-bound NA rule; the E3 summaries and this "
                  "report suppress slope/intercept when fewer than 15 repeats are valid."]
    else:
        lines += ["\n### Confirmatory comparisons", "",
                  "Unavailable: the saved E2 or main E3/E4 table is missing."]
    e5 = read_table(tables, "e5_few_label.tsv")
    lines += ["\n### E5 (exploratory)", ""]
    if e5 is None:
        lines.append("Results unavailable.")
    else:
        rows = []
        for _, row in e5.iterrows():
            record = [
                str(row["setting"]), str(int(row["k"])), str(row["method"]),
            ]
            for metric in ("brier", "calibration_intercept", "calibration_slope"):
                for stem in (metric, f"{metric}_difference_vs_none"):
                    record.append(
                        f"{number(row[stem + '_median'])} "
                        f"[{number(row[stem + '_q1'])}, {number(row[stem + '_q3'])}]"
                    )
            record.append(number(row["fraction_fit_failed"]))
            rows.append(record)
        lines.append(rows_table(
            ["Setting", "k", "Method", "Brier [Q1, Q3]", "Paired Δ Brier",
             "Intercept [Q1, Q3]", "Paired Δ intercept", "Slope [Q1, Q3]",
             "Paired Δ slope", "Fit failure fraction"], rows
        ))
        lines += ["", "The A0 recalibration-is-ill-posed warning was a pre-result "
                  "hypothesis; observed recalibrated A0 predictions did not remain at "
                  "the clip bounds. This exploratory observation does not establish generalizability."]
    lines += [
        "\n### E7 / secondary-label DNMT3A analysis (exploratory)",
        "",
        "E7 secondary-label cross-cohort results are stored in the E3/E4 "
        "`e3_e4_cross_cohort*.tsv` tables (`outcome == DNMT3A`), not in a "
        "separate E7 file. This descriptive summary is limited to the "
        "preregistered primary no-adaptation setting (A0, logistic, no "
        "calibrator); DNMT3A analyses are exploratory.",
        "",
    ]
    dnmt3a_rows = []
    for label, suffix in (("Main", ""), ("E6a", "_e6a"), ("E6b", "_e6b")):
        frame = read_table(tables, f"e3_e4_cross_cohort{suffix}.tsv")
        counts = read_table(tables, f"analysis_set{suffix}.tsv")
        if frame is None:
            continue
        rows = select(
            frame,
            split_set="registered_main_split",
            method="A0",
            model="logistic",
            calibrator="none",
            outcome="DNMT3A",
        )
        if len(rows) != 1:
            raise ValueError(f"{label} DNMT3A A0 primary row is missing or duplicated")
        row = rows.iloc[0]
        positives = "NA"
        if counts is not None:
            target_count = counts.loc[counts["cohort"] == "tcga_laml"]
            if len(target_count) == 1:
                positives = (
                    f"{int(target_count.iloc[0]['dnmt3a_positives'])}/"
                    f"{int(target_count.iloc[0]['dnmt3a_profiled_samples'] if 'dnmt3a_profiled_samples' in target_count else target_count.iloc[0]['n_samples'])}"
                )
        dnmt3a_rows.append([
            label,
            positives,
            str(row["status"]),
            interval(row, "target_auroc"),
            interval(row, "target_brier"),
            interval(row, "target_calibration_intercept"),
            interval(row, "target_calibration_slope"),
            (
                f"{number(row['source_repeat_median_brier'])} "
                f"[{number(row['source_repeat_q1_brier'])}, "
                f"{number(row['source_repeat_q3_brier'])}]"
            ),
            (
                f"{number(row['source_repeat_median_calibration_intercept'])} "
                f"[{number(row['source_repeat_q1_calibration_intercept'])}, "
                f"{number(row['source_repeat_q3_calibration_intercept'])}]"
            ),
            (
                f"{number(row['source_repeat_median_calibration_slope'])} "
                f"[{number(row['source_repeat_q1_calibration_slope'])}, "
                f"{number(row['source_repeat_q3_calibration_slope'])}]"
            ),
            number(row["source_test_brier"]),
            number(row["source_test_calibration_intercept"]),
            number(row["source_test_calibration_slope"]),
        ])
    if dnmt3a_rows:
        lines.append(rows_table(
            [
                "Analysis set", "Target DNMT3A positives/profiled", "Status",
                "AUROC (target 95% CI)", "Brier (target 95% CI)",
                "Intercept (target 95% CI)", "Slope (target 95% CI)",
                "Source-repeat Brier median [Q1, Q3]",
                "Source-repeat intercept median [Q1, Q3]",
                "Source-repeat slope median [Q1, Q3]",
                "Source main-split Brier",
                "Source main-split intercept",
                "Source main-split slope",
            ],
            dnmt3a_rows,
        ))
        lines += [
            "",
            "Source-repeat columns are the saved E3/E4 source-test summaries "
            "over 20 repeats. Source main-split metrics are shown separately. Target "
            "intervals are the existing patient-bootstrap intervals. This "
            "table summarizes results only and does not add a confirmatory comparison.",
        ]
    else:
        lines.append("DNMT3A cross-cohort result tables are unavailable.")
    coverage = read_table(tables, "conformal_coverage.tsv")
    lines += ["\n### Split conformal (exploratory)", ""]
    if coverage is None:
        lines.append("Results unavailable.")
    else:
        coverage = select(coverage, split_set="registered_main_split", model="logistic",
                          calibrator="none", outcome="NPM1", status="complete")
        rows = [[str(r["method"]), number(r["source_test_coverage"]),
                 interval(r, "target_coverage"), number(r["target_positive_class_coverage"]),
                 number(r["target_negative_class_coverage"]),
                 number(r["target_mean_prediction_set_size"]),
                 number(r["target_empty_set_fraction"])] for _, r in coverage.iterrows()]
        lines.append(rows_table(["Setting", "Source coverage", "Target coverage (95% CI)",
                                 "Target positive", "Target negative", "Mean set size",
                                 "Empty fraction"], rows))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    text = args.readme.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one ordered pair of results markers")
    before, remainder = text.split(START)
    _, after = remainder.split(END)
    args.readme.write_text(
        before + START + "\n" + report(args.tables) + "\n" + END + after,
        encoding="utf-8",
    )
    print(f"Updated {args.readme} from {args.tables}")


if __name__ == "__main__":
    main()
