# Scenario 3 Trial-1000 Summary

## 1. Purpose

This document summarizes the controlled 1000-trial Scenario 3 confirmation run for the Z5 array using the selected ALSS configuration.

This run is intended to confirm whether the previously archived 500-trial trend remains stable with a larger Monte Carlo sample size.

Selected ALSS configuration:

```text
alss_mode  = ar1
alss_tau   = 0.25
alss_coreL = 3
```

---

## 2. Experiment Configuration

```text
Scenario: 3
Array: Z5
Trials: 1000
ALSS mode: ar1
ALSS tau: 0.25
ALSS coreL: 3
Coupling levels: c1 = 0.0 and c1 = 0.3
```

Coupling interpretation:

```text
c1 = 0.0 means no mutual coupling.
c1 = 0.3 means mutual coupling is enabled.
```

---

## 3. Execution Command

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$(Get-Location)"

python core\analysis_scripts\run_paper_experiments.py `
    --scenario 3 `
    --trials 1000 `
    --arrays Z5 `
    --alss-mode ar1 `
    --alss-tau 0.25 `
    --alss-core-l 3 `
    --output-dir "$env:TEMP\mimo_scenario3_ar1_tau025_trial1000"
```

---

## 4. Temporary Output Location

```text
$env:TEMP\mimo_scenario3_ar1_tau025_trial1000\scenario3_alss_effectiveness.csv
```

The output was first generated outside the repository and inspected before archival.

---

## 5. Expected Repository Result File

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

---

## 6. High-Level Result

The 1000-trial result confirms the main trend observed in the 500-trial run.

Summary over reported rows:

```text
Reported rows: 16
Mean improvement: approximately +9.85%
Median improvement: approximately +8.66%
Worst improvement: approximately -0.50%
Best improvement: approximately +33.79%
Positive rows: 15 / 16
```

Summary over unique conditions:

```text
Unique conditions: 14
Mean improvement: approximately +9.76%
Median improvement: approximately +7.73%
Worst improvement: approximately -0.50%
Best improvement: approximately +33.79%
Positive unique conditions: 13 / 14
```

---

## 7. Coupling-Level Summary

No-coupling case:

```text
Coupling: c1 = 0.0
Unique conditions: 7
Mean improvement: approximately +5.60%
Median improvement: approximately +5.54%
Worst improvement: approximately +0.91%
Best improvement: approximately +10.98%
Positive conditions: 7 / 7
```

Mutual-coupling case:

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

## 8. Comparison With Trial-500 Result

The 1000-trial run is slightly more conservative than the 500-trial run, but it confirms the same scientific trend.

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
The 1000-trial result confirms the 500-trial trend.
ALSS remains more beneficial under mutual coupling than under no-coupling conditions.
The worst-case degradation remains small.
The majority of reported and unique conditions remain positive.
```

---

## 9. Scientific Interpretation

The result supports the hypothesis that ALSS provides a useful post-geometry statistical regularization layer for the Z5 sparse array.

The main observation is that the mean improvement is stronger when mutual coupling is enabled:

```text
No coupling mean improvement:       approximately +5.60%
Mutual coupling mean improvement:   approximately +13.92%
```

This is consistent with the current research narrative:

```text
Weight-constrained sparse-array geometry reduces coupling-related bias.
ALSS reduces finite-snapshot lag-estimation variance.
For Z5, the combined effect is strongest under mutual coupling and moderate-to-high SNR.
```

---

## 10. Conservative Claim Supported

Recommended claim:

```text
For the Z5 sparse array, ALSS with ar1/tau=0.25/coreL=3 consistently improves Scenario 3 coarray-MUSIC RMSE in most tested conditions, with stronger gains under mutual coupling than under no-coupling conditions.
```

Avoid claiming:

```text
ALSS always improves performance.
ALSS is universally optimal for all sparse arrays.
ALSS is harmless in every trial.
The Z5 Scenario 3 result proves paper-wide performance across all geometries.
```

---

## 11. Current Status

```text
Scenario 3 trial-1000 run: completed
Result trend: confirms trial-500
Git status after run: clean
Next step: archive trial-1000 CSV if not already archived
```

Recommended next repository action:

```text
1. Archive the trial-1000 CSV under results/paper_experiments.
2. Commit this summary document.
3. Generate trial-1000 figures only after the archived CSV is merged.
4. Decide whether the paper should use trial-1000 as the primary result and trial-500 as validation history.
```