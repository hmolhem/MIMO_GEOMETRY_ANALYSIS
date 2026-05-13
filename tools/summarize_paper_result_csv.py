"""
Summarize paper experiment result CSV files.

This tool is intended for archived paper-result CSV files such as:

    results/paper_experiments/scenario3_z5_ar1_tau025_trial500.csv

It computes compact, paper-friendly summary metrics without modifying the input file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "RMSE_Baseline",
    "RMSE_ALSS",
    "Improvement_%",
    "P_Value",
    "Harmlessness_%",
    "Trials",
}

CONDITION_COLUMNS = ["Array", "Coupling_c1", "SNR_dB", "Snapshots"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paper experiment result CSV files."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input paper-result CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output CSV path for summary metrics.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional output Markdown path for a compact report.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for p-values. Default: 0.05.",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame, input_path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{input_path} is missing required columns: {missing_text}")

    if df.empty:
        raise ValueError(f"{input_path} is empty.")


def prepare_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    numeric_columns = [
        "RMSE_Baseline",
        "RMSE_ALSS",
        "Improvement_%",
        "P_Value",
        "Harmlessness_%",
        "Trials",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    return working.dropna(subset=numeric_columns)


def unique_conditions(df: pd.DataFrame) -> pd.DataFrame:
    available_condition_columns = [
        column for column in CONDITION_COLUMNS if column in df.columns
    ]

    if not available_condition_columns:
        return df.copy()

    return df.drop_duplicates(subset=available_condition_columns).copy()


def summarize_frame(
    df: pd.DataFrame,
    scope: str,
    alpha: float,
) -> dict[str, float | int | str]:
    improvements = df["Improvement_%"]
    p_values = df["P_Value"]
    harmlessness = df["Harmlessness_%"]

    positive_mask = improvements > 0
    significant_mask = p_values < alpha

    return {
        "Scope": scope,
        "Rows": int(len(df)),
        "Trials_Min": int(df["Trials"].min()),
        "Trials_Max": int(df["Trials"].max()),
        "Mean_Improvement_Pct": float(improvements.mean()),
        "Median_Improvement_Pct": float(improvements.median()),
        "Worst_Improvement_Pct": float(improvements.min()),
        "Best_Improvement_Pct": float(improvements.max()),
        "Positive_Rows": int(positive_mask.sum()),
        "Positive_Rate_Pct": float(100.0 * positive_mask.mean()),
        "Significant_Rows": int(significant_mask.sum()),
        "Significant_Positive_Rows": int((positive_mask & significant_mask).sum()),
        "Mean_Harmlessness_Pct": float(harmlessness.mean()),
        "Worst_Harmlessness_Pct": float(harmlessness.min()),
        "Mean_RMSE_Baseline": float(df["RMSE_Baseline"].mean()),
        "Mean_RMSE_ALSS": float(df["RMSE_ALSS"].mean()),
    }


def build_summary_table(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    unique_df = unique_conditions(df)

    rows: list[dict[str, float | int | str]] = [
        summarize_frame(df, "reported_rows", alpha),
        summarize_frame(unique_df, "unique_conditions", alpha),
    ]

    if "Coupling_c1" in unique_df.columns:
        for coupling_value, group in unique_df.groupby("Coupling_c1", sort=True):
            rows.append(
                summarize_frame(
                    group,
                    f"unique_conditions_coupling_c1={coupling_value}",
                    alpha,
                )
            )

    return pd.DataFrame(rows)


def find_extreme_rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    worst_row = df.loc[df["Improvement_%"].idxmin()]
    best_row = df.loc[df["Improvement_%"].idxmax()]
    return worst_row, best_row


def format_condition(row: pd.Series) -> str:
    parts = []
    for column in CONDITION_COLUMNS:
        if column in row.index:
            parts.append(f"{column}={row[column]}")
    return ", ".join(parts)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()

    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")

    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display_df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_markdown_report(
    input_path: Path,
    summary_df: pd.DataFrame,
    unique_df: pd.DataFrame,
) -> str:
    worst_row, best_row = find_extreme_rows(unique_df)

    report = [
        "# Paper Result CSV Summary",
        "",
        f"Input file: `{input_path}`",
        "",
        "## Summary Metrics",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Worst Condition",
        "",
        f"- Condition: `{format_condition(worst_row)}`",
        f"- Improvement: `{worst_row['Improvement_%']:.3f}%`",
        f"- p-value: `{worst_row['P_Value']:.6g}`",
        f"- Harmlessness: `{worst_row['Harmlessness_%']:.3f}%`",
        "",
        "## Best Condition",
        "",
        f"- Condition: `{format_condition(best_row)}`",
        f"- Improvement: `{best_row['Improvement_%']:.3f}%`",
        f"- p-value: `{best_row['P_Value']:.6g}`",
        f"- Harmlessness: `{best_row['Harmlessness_%']:.3f}%`",
        "",
        "## Interpretation Reminder",
        "",
        "Use this summary as an analysis aid. It does not replace the raw CSV.",
        "Check duplicated plot points before using aggregate means in a paper.",
        "",
    ]

    return "\n".join(report)


def print_console_summary(summary_df: pd.DataFrame, unique_df: pd.DataFrame) -> None:
    print("\nPaper-result summary:")
    print(summary_df.to_string(index=False))

    worst_row, best_row = find_extreme_rows(unique_df)

    print("\nWorst unique condition:")
    print(f"  {format_condition(worst_row)}")
    print(f"  Improvement = {worst_row['Improvement_%']:.3f}%")
    print(f"  p-value     = {worst_row['P_Value']:.6g}")
    print(f"  harmless    = {worst_row['Harmlessness_%']:.3f}%")

    print("\nBest unique condition:")
    print(f"  {format_condition(best_row)}")
    print(f"  Improvement = {best_row['Improvement_%']:.3f}%")
    print(f"  p-value     = {best_row['P_Value']:.6g}")
    print(f"  harmless    = {best_row['Harmlessness_%']:.3f}%")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df, input_path)

    working = prepare_numeric_columns(df)
    unique_df = unique_conditions(working)
    summary_df = build_summary_table(working, args.alpha)

    print_console_summary(summary_df, unique_df)

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSaved summary CSV: {output_csv}")

    if args.output_md:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            build_markdown_report(input_path, summary_df, unique_df),
            encoding="utf-8",
        )
        print(f"Saved summary Markdown: {output_md}")


if __name__ == "__main__":
    main()