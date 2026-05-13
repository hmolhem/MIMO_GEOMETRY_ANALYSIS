# Scenario 3 Trial-500 Figure Index

This document maps the archived Scenario 3 trial-500 result figures to their source data, plotting tool, experiment configuration, and paper-facing scientific claims.

---

## 1. Source Data

The figures in this index were generated from the archived Scenario 3 Z5 ALSS trial-500 CSV result:

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial500.csv
```

The result corresponds to:

```text
Scenario: 3
Array: Z5
Trials: 500
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
results/figures/scenario3_trial500/
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
| Fig. 5 | `scenario3_harmlessness_vs_snr.png` | Show trial-level harmlessness versus SNR | ALSS improves mean RMSE in most conditions, but is not harmless in every trial. |

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

The 500-trial result showed:

```text
Reported rows: 16
Unique conditions: 14
Mean improvement over reported rows: approximately +10.52%
Mean improvement over unique conditions: approximately +10.54%
Worst improvement: approximately -0.78%
Best improvement: approximately +37.56%
Coupled-case mean improvement: approximately +15.20%
Non-coupled mean improvement: approximately +5.88%
```

Worst condition:

```text
Array: Z5
Coupling: c1=0.3
SNR: -5 dB
Snapshots: 64
Improvement: approximately -0.78%
p-value: 0.4403
```

Best condition:

```text
Array: Z5
Coupling: c1=0.3
SNR: 15 dB
Snapshots: 64
Improvement: approximately +37.56%
p-value: 2.50e-06
```

---

## 7. Conservative Scientific Claim

Recommended paper claim:

```text
For the Z5 sparse array, ALSS with ar1/tau=0.25/coreL=3 improves coarray-MUSIC RMSE in most tested Scenario 3 conditions, with the strongest gains under mutual coupling and moderate-to-high SNR.
```

Avoid claiming:

```text
ALSS always improves performance.
ALSS is universally optimal for all arrays.
ALSS is harmless in every trial.
The Scenario 3 Z5 result proves paper-wide performance across all geometries.
```

---

## 8. Next Step

The next recommended validation step is a controlled 1000-trial confirmation run for Scenario 3 / Z5 / ar1 / tau=0.25 / coreL=3.

This should be run after the current figure index is merged into `main`.