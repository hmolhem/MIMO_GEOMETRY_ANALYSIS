# Scenario 3 Trial-1000 Figure Index

This document maps the archived Scenario 3 trial-1000 result figures to their source data, plotting tool, experiment configuration, and paper-facing scientific claims.

---

## 1. Source Data

The figures in this index were generated from the archived Scenario 3 Z5 ALSS trial-1000 CSV result:

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

The result corresponds to:

```text
Scenario: 3
Array: Z5
Trials: 1000
ALSS mode: ar1
ALSS tau: 0.25
ALSS coreL: 3
Coupling levels: c1=0.0 and c1=0.3
```

Coupling interpretation:

```text
c1 = 0.0 means no mutual coupling.
c1 = 0.3 means mutual coupling is enabled.
```

---

## 2. Plotting Tool

The figures were generated using:

```text
tools/plot_scenario3_results.py
```

The plotting tool removes duplicate Scenario 3 overlap points before plotting. In Scenario 3, the condition `SNR=5 dB, Snapshots=64` appears in both the SNR sweep and the snapshot sweep.

---

## 3. Figure Directory

Archived figures are stored in:

```text
results/figures/scenario3_trial1000/
```

Directory contents:

```text
README.md
scenario3_rmse_vs_snr_c1_0p0.png
scenario3_rmse_vs_snr_c1_0p3.png
scenario3_improvement_vs_snr.png
scenario3_improvement_vs_snapshots.png
scenario3_harmlessness_vs_snr.png
```

---

## 4. Paper-Facing Figure Map

| Paper Figure Candidate | Repository File | Purpose | Main Claim Supported |
|---|---|---|---|
| Fig. 1 | `scenario3_rmse_vs_snr_c1_0p0.png` | Compare baseline and ALSS RMSE versus SNR without mutual coupling | ALSS provides moderate RMSE improvement for Z5 even without coupling. |
| Fig. 2 | `scenario3_rmse_vs_snr_c1_0p3.png` | Compare baseline and ALSS RMSE versus SNR with mutual coupling enabled | ALSS provides stronger RMSE reduction when mutual coupling is present. |
| Fig. 3 | `scenario3_improvement_vs_snr.png` | Show percentage RMSE improvement versus SNR for both coupling cases | Improvement is strongest under mutual coupling and moderate-to-high SNR. |
| Fig. 4 | `scenario3_improvement_vs_snapshots.png` | Show percentage RMSE improvement versus snapshot count at fixed SNR = 5 dB | ALSS improvement remains positive across snapshot counts, especially under coupling. |
| Fig. 5 | `scenario3_harmlessness_vs_snr.png` | Show trial-level harmlessness versus SNR | ALSS improves mean RMSE in most conditions, but it is not harmless in every trial. |

---

## 5. Recommended Figures for the First Paper Draft

For a compact IEEE-style results section, the strongest first-draft figure set is:

```text
Fig. 1: scenario3_improvement_vs_snr.png
Fig. 2: scenario3_rmse_vs_snr_c1_0p3.png
Fig. 3: scenario3_improvement_vs_snapshots.png
```

Rationale:

```text
1. Improvement vs SNR directly communicates the ALSS benefit.
2. RMSE vs SNR under mutual coupling shows the most important practical case.
3. Improvement vs snapshots shows finite-snapshot robustness.
```

The harmlessness figure is useful for discussion or supplementary material because it supports a conservative claim: ALSS improves average RMSE in most tested conditions, but it is not guaranteed to improve every individual trial.

---

## 6. Main Numerical Summary

The 1000-trial result showed:

```text
Reported rows: 16
Unique conditions: 14
Mean improvement over reported rows: approximately +9.85%
Mean improvement over unique conditions: approximately +9.76%
Worst improvement: approximately -0.50%
Best improvement: approximately +33.79%
Positive reported rows: 15 / 16
Positive unique conditions: 13 / 14
Coupled-case mean improvement: approximately +13.92%
Non-coupled mean improvement: approximately +5.60%
```

No-coupling summary:

```text
Coupling: c1 = 0.0
Unique conditions: 7
Mean improvement: approximately +5.60%
Median improvement: approximately +5.54%
Worst improvement: approximately +0.91%
Best improvement: approximately +10.98%
Positive conditions: 7 / 7
```

Mutual-coupling summary:

```text
Coupling: c1 = 0.3
Unique conditions: 7
Mean improvement: approximately +13.92%
Median improvement: approximately +11.47%
Worst improvement: approximately -0.50%
Best improvement: approximately +33.79%
Positive conditions: 6 / 7
```

---

## 7. Comparison With Trial-500

The 1000-trial result confirms the same trend observed in the 500-trial result.

```text
Trial-500 reported mean improvement:   approximately +10.52%
Trial-1000 reported mean improvement:  approximately +9.85%

Trial-500 coupled mean improvement:    approximately +15.20%
Trial-1000 coupled mean improvement:   approximately +13.92%

Trial-500 worst improvement:           approximately -0.78%
Trial-1000 worst improvement:          approximately -0.50%

Trial-500 best improvement:            approximately +37.56%
Trial-1000 best improvement:           approximately +33.79%
```

Interpretation:

```text
The trial-1000 result is slightly more conservative than trial-500, but it confirms the same scientific trend.
ALSS remains more beneficial under mutual coupling than under no-coupling conditions.
The worst-case degradation remains small.
Most reported and unique conditions remain positive.
```

---

## 8. Conservative Scientific Claim

Recommended paper claim:

```text
For the Z5 sparse array, ALSS with ar1/tau=0.25/coreL=3 consistently improves Scenario 3 coarray-MUSIC RMSE in most tested conditions, with stronger gains under mutual coupling than under no-coupling conditions.
```

Avoid claiming:

```text
ALSS always improves performance.
ALSS is universally optimal for all arrays.
ALSS is harmless in every trial.
The Z5 Scenario 3 result proves paper-wide performance across all geometries.
```

---

## 9. Relationship to Trial-500 Figures

The trial-500 figures remain useful as an earlier validation artifact, but the trial-1000 figures should be treated as the stronger confirmation set for paper drafting.

Recommended usage:

```text
Use trial-1000 figures as the primary paper figures.
Keep trial-500 figures as reproducibility history and intermediate validation evidence.
```

---

## 10. Next Step

The next recommended step is to update the paper-facing experiment summary and begin integrating the trial-1000 figures into the IEEE-style paper draft.

Potential next files:

```text
docs/experiment_summary.md
papers/alss_ieee_paper.tex
```

Before editing the paper, confirm that the repository is clean and that this figure index has been merged into main.