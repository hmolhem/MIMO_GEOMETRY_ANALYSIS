"""
Plot Scenario 3 paper experiment results.

This tool reads a Scenario 3 CSV result file, such as:

    results/paper_experiments/scenario3_z5_ar1_tau025_trial500.csv

and generates publication-oriented diagnostic plots.

Generated plots:
    1. RMSE vs SNR for each coupling level
    2. Improvement vs SNR
    3. Improvement vs snapshots
    4. Harmlessness vs SNR

The tool does not modify the input CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {
    "Array",
    "Coupling_c1",
    "SNR_dB",
    "Snapshots",
    "RMSE_Baseline",
    "RMSE_ALSS",
    "Improvement_%",
    "Harmlessness_%",
    "P_Value",
    "Trials",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots from Scenario 3 paper-result CSV files."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input Scenario 3 result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where PNG plots will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI. Default: 300.",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame, input_path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{input_path} is missing required columns: {missing_text}")

    if df.empty:
        raise ValueError(f"{input_path} is empty.")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    numeric_columns = [
        "Coupling_c1",
        "SNR_dB",
        "Snapshots",
        "RMSE_Baseline",
        "RMSE_ALSS",
        "Improvement_%",
        "Harmlessness_%",
        "P_Value",
        "Trials",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.dropna(subset=numeric_columns)

    # Scenario 3 contains duplicated SNR=5, M=64 entries because the same point
    # appears in both the SNR sweep and snapshot sweep. Keep one copy for plots.
    working = working.drop_duplicates(
        subset=["Array", "Coupling_c1", "SNR_dB", "Snapshots"],
        keep="first",
    )

    return working.sort_values(["Coupling_c1", "Snapshots", "SNR_dB"])


def safe_coupling_label(coupling_value: float) -> str:
    return f"c1={coupling_value:g}"


def safe_filename_coupling(coupling_value: float) -> str:
    return f"c1_{str(coupling_value).replace('.', 'p')}"


def save_current_figure(output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_rmse_vs_snr(df: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    output_paths: list[Path] = []

    snr_df = df[df["Snapshots"] == 64].copy()

    for coupling_value, group in snr_df.groupby("Coupling_c1", sort=True):
        group = group.sort_values("SNR_dB")

        plt.figure(figsize=(7.0, 4.5))
        plt.plot(
            group["SNR_dB"],
            group["RMSE_Baseline"],
            marker="o",
            label="Baseline",
        )
        plt.plot(
            group["SNR_dB"],
            group["RMSE_ALSS"],
            marker="s",
            label="ALSS",
        )
        plt.xlabel("SNR (dB)")
        plt.ylabel("RMSE (degrees)")
        plt.title(f"Scenario 3 Z5 RMSE vs SNR ({safe_coupling_label(coupling_value)})")
        plt.grid(True, alpha=0.3)
        plt.legend()

        output_path = output_dir / f"scenario3_rmse_vs_snr_{safe_filename_coupling(coupling_value)}.png"
        save_current_figure(output_path, dpi)
        output_paths.append(output_path)

    return output_paths


def plot_improvement_vs_snr(df: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    snr_df = df[df["Snapshots"] == 64].copy()

    plt.figure(figsize=(7.0, 4.5))

    for coupling_value, group in snr_df.groupby("Coupling_c1", sort=True):
        group = group.sort_values("SNR_dB")
        plt.plot(
            group["SNR_dB"],
            group["Improvement_%"],
            marker="o",
            label=safe_coupling_label(coupling_value),
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE improvement (%)")
    plt.title("Scenario 3 Z5 ALSS Improvement vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Coupling")

    output_path = output_dir / "scenario3_improvement_vs_snr.png"
    save_current_figure(output_path, dpi)
    return output_path


def plot_improvement_vs_snapshots(df: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    snapshot_df = df[df["SNR_dB"] == 5].copy()

    plt.figure(figsize=(7.0, 4.5))

    for coupling_value, group in snapshot_df.groupby("Coupling_c1", sort=True):
        group = group.sort_values("Snapshots")
        plt.plot(
            group["Snapshots"],
            group["Improvement_%"],
            marker="o",
            label=safe_coupling_label(coupling_value),
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Snapshots")
    plt.ylabel("RMSE improvement (%)")
    plt.title("Scenario 3 Z5 ALSS Improvement vs Snapshots")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Coupling")

    output_path = output_dir / "scenario3_improvement_vs_snapshots.png"
    save_current_figure(output_path, dpi)
    return output_path


def plot_harmlessness_vs_snr(df: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    snr_df = df[df["Snapshots"] == 64].copy()

    plt.figure(figsize=(7.0, 4.5))

    for coupling_value, group in snr_df.groupby("Coupling_c1", sort=True):
        group = group.sort_values("SNR_dB")
        plt.plot(
            group["SNR_dB"],
            group["Harmlessness_%"],
            marker="o",
            label=safe_coupling_label(coupling_value),
        )

    plt.axhline(50.0, linestyle="--", linewidth=1.0)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Harmlessness (%)")
    plt.title("Scenario 3 Z5 ALSS Harmlessness vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Coupling")

    output_path = output_dir / "scenario3_harmlessness_vs_snr.png"
    save_current_figure(output_path, dpi)
    return output_path


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df, input_path)

    working = prepare_dataframe(df)

    if working.empty:
        raise ValueError("No valid numeric rows remain after preprocessing.")

    output_paths: list[Path] = []
    output_paths.extend(plot_rmse_vs_snr(working, output_dir, args.dpi))
    output_paths.append(plot_improvement_vs_snr(working, output_dir, args.dpi))
    output_paths.append(plot_improvement_vs_snapshots(working, output_dir, args.dpi))
    output_paths.append(plot_harmlessness_vs_snr(working, output_dir, args.dpi))

    print("\nGenerated Scenario 3 plot files:")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()