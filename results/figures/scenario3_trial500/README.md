# Scenario 3 Trial-500 Figures

This directory contains publication-oriented figures generated from the controlled Scenario 3 Z5 ALSS trial-500 result.

## Source CSV

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial500.csv
```

## Plotting Tool

```text
tools/plot_scenario3_results.py
```

## Experiment Configuration

```text
Scenario: 3
Array: Z5
Trials: 500
ALSS mode: ar1
ALSS tau: 0.25
ALSS coreL: 3
Coupling levels: c1=0.0 and c1=0.3
```

## Coupling Interpretation

```text
c1 = 0.0 means no mutual coupling.
c1 = 0.3 means mutual coupling is enabled.
```

## Generated Figures

```text
scenario3_rmse_vs_snr_c1_0p0.png
scenario3_rmse_vs_snr_c1_0p3.png
scenario3_improvement_vs_snr.png
scenario3_improvement_vs_snapshots.png
scenario3_harmlessness_vs_snr.png
```

## Notes

The plot labels distinguish between:

```text
No coupling (c1=0)
Mutual coupling (c1=0.3)
```

These figures are generated artifacts from the archived 500-trial CSV result.