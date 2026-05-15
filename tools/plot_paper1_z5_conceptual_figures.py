"""Generate Paper 1 conceptual figures for the Scenario 3 Z5 study.

This script intentionally uses the same canonical Z5 N=7 grid positions
used by the Scenario 3 paper experiments:

    [0, 5, 8, 11, 14, 17, 21]

Outputs:
    results/figures/paper1_conceptual/z5_sensor_geometry.png
    results/figures/paper1_conceptual/z5_coarray_weights.png
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Z5_POSITIONS = np.array([0, 5, 8, 11, 14, 17, 21], dtype=int)


def compute_coarray_weights(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted lags and weights for the full difference coarray."""
    diffs = []

    for ni in positions:
        for nj in positions:
            diffs.append(int(ni - nj))

    counts = Counter(diffs)
    lags = np.array(sorted(counts.keys()), dtype=int)
    weights = np.array([counts[int(lag)] for lag in lags], dtype=int)

    return lags, weights


def plot_sensor_geometry(output_dir: Path) -> None:
    """Plot physical Z5 sensor locations."""
    fig, ax = plt.subplots(figsize=(8, 1.8))

    ax.scatter(
        Z5_POSITIONS,
        np.zeros_like(Z5_POSITIONS),
        marker="s",
        s=90,
    )

    for idx, pos in enumerate(Z5_POSITIONS):
        ax.text(
            pos,
            0.08,
            f"S{idx + 1}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_yticks([])
    ax.set_xlabel("Sensor position (half-wavelength grid units)")
    ax.set_title("Z5 Sparse Array Geometry (N=7)")
    ax.set_xlim(Z5_POSITIONS.min() - 1, Z5_POSITIONS.max() + 1)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "z5_sensor_geometry.png", dpi=300)
    plt.close(fig)


def plot_coarray_weights(output_dir: Path) -> None:
    """Plot Z5 difference-coarray weights."""
    lags, weights = compute_coarray_weights(Z5_POSITIONS)

    fig, ax = plt.subplots(figsize=(8, 3.8))

    markerline, stemlines, baseline = ax.stem(lags, weights)
    plt.setp(markerline, markersize=4)
    plt.setp(stemlines, linewidth=1)
    plt.setp(baseline, linewidth=0.8)

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("Difference coarray lag")
    ax.set_ylabel("Weight w(l)")
    ax.set_title("Z5 Difference-Coarray Weights (N=7)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Highlight the missing critical small lags on the positive side.
    for lag in [1, 2]:
        ax.text(
            lag,
            0.25,
            f"w({lag})=0",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "z5_coarray_weights.png", dpi=300)
    plt.close(fig)


def main() -> None:
    output_dir = Path("results/figures/paper1_conceptual")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_sensor_geometry(output_dir)
    plot_coarray_weights(output_dir)

    lags, weights = compute_coarray_weights(Z5_POSITIONS)
    lag_set = set(int(lag) for lag in lags)

    print("Paper 1 conceptual figures generated.")
    print(f"Output directory: {output_dir}")
    print(f"Z5 positions: {Z5_POSITIONS.tolist()}")
    print("Positive small-lag weights:")

    for lag in [1, 2, 3, 4, 5]:
        if lag in lag_set:
            weight = int(weights[np.where(lags == lag)][0])
        else:
            weight = 0

        print(f"  w({lag}) = {weight}")


if __name__ == "__main__":
    main()