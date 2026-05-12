# ALSS Parameter-Selection Decision

## 1. Purpose

This document records the current ALSS parameter-selection decision for the `MIMO_GEOMETRY_ANALYSIS` project.

The decision is based on diagnostic calibration runs performed after the clean rebuild process.

This document does not claim final paper-ready results. It only records the calibration decision that should be used before updating the final paper experiment runner and before running high-trial paper experiments.

---

## 2. Decision Summary

The selected ALSS calibration candidate is:

```text
alss_mode  = ar1
alss_tau   = 0.25
alss_coreL = 3
```

Decision status:

```text
Frozen for the current Z5 calibration protocol.
Not yet validated as final paper-wide setting.
```

This setting should be used as the next candidate when updating the paper experiment runner.

---

## 3. Calibration Context

The diagnostic calibration was performed using:

```text
Array: Z5
True DOAs: [15.0, -20.0] degrees
SNR values: 5, 10, 15 dB
Coupling values: 0.0, 0.3
Snapshots: 64
Trials: 100
Modes tested: zero, ar1
Tau values tested: 0.1, 0.25, 0.5, 1.0
CoreL: 3
```

The diagnostic run used:

```text
tools/diagnose_alss_sweep.py
```

The summary run used:

```text
tools/summarize_alss_diagnostics.py
```

---

## 4. Diagnostic Command

The 100-trial diagnostic sweep was run with:

```powershell
python tools\diagnose_alss_sweep.py `
    --trials 100 `
    --array Z5 `
    --snrs 5 10 15 `
    --couplings 0.0 0.3 `
    --snapshots 64 `
    --modes zero ar1 `
    --taus 0.1 0.25 0.5 1.0 `
    --output-dir "$env:TEMP\mimo_alss_diag_trial100"
```

The diagnostic CSV was written to:

```text
$env:TEMP\mimo_alss_diag_trial100\alss_mode_tau_diagnostic.csv
```

---

## 5. Summary Command

The diagnostic summary was generated with:

```powershell
python tools\summarize_alss_diagnostics.py `
    --input "$env:TEMP\mimo_alss_diag_trial100\alss_mode_tau_diagnostic.csv" `
    --output "$env:TEMP\mimo_alss_diag_trial100\alss_selection_summary.csv"
```

The summary CSV was written to:

```text
$env:TEMP\mimo_alss_diag_trial100\alss_selection_summary.csv
```

---

## 6. Selection Result

The summary tool selected:

```text
mode = ar1
tau  = 0.25
```

with the following calibration metrics:

```text
Mean improvement       = +16.666%
Worst improvement      = +3.585%
Average harmlessness   = 76.000%
Positive improvement   = 100%
Pass                   = True
```

Interpretation:

```text
The selected candidate improved RMSE in all six calibration conditions.
The worst-case improvement was still positive.
The average harmlessness exceeded the protocol threshold.
```

---

## 7. Comparison With Other Candidates

The diagnostic summary showed that `zero, tau=0.1` had a higher mean improvement:

```text
zero, tau=0.1
Mean improvement = +33.910%
```

However, this candidate was rejected because:

```text
Average harmlessness = 40.500%
```

This means it improved mean RMSE but was less reliable on a trial-by-trial basis.

Therefore, `zero, tau=0.1` is considered too risky for the current paper workflow.

The selected candidate `ar1, tau=0.25` is more conservative and more defensible because it balances:

```text
positive mean improvement
positive worst-case improvement
acceptable harmlessness
consistent improvement across calibration conditions
```

---

## 8. Decision Rationale

The selected candidate is preferred because:

```text
1. It passed the robustness filters.
2. It had positive improvement in all calibration conditions.
3. It avoided the severe degradation seen in some zero-mode settings.
4. It was consistent across coupling and non-coupling cases.
5. It did not rely on a high-risk low-harmlessness improvement pattern.
```

This supports using:

```text
ar1, tau=0.25, coreL=3
```

as the next frozen ALSS configuration for controlled evaluation.

---

## 9. Scientific Limitation

This decision is based on a calibration grid, not on final paper-wide experiments.

The current decision is valid only for:

```text
Z5 calibration
SNR = 5, 10, 15 dB
Coupling c1 = 0.0, 0.3
Snapshots = 64
Trials = 100
```

It should not yet be generalized to all arrays, all snapshot counts, or all paper figures without further validation.

---

## 10. Required Next Engineering Step

The paper experiment runner currently uses a fixed ALSS setting internally.

The runner must be updated to expose the following command-line arguments:

```text
--alss-mode
--alss-tau
--alss-core-l
```

The next candidate final-run setting should be passed explicitly as:

```powershell
--alss-mode ar1 --alss-tau 0.25 --alss-core-l 3
```

---

## 11. Required Validation Before Final Paper Run

Before running final 500-trial or 1000-trial paper experiments, perform a controlled validation using the updated runner.

Recommended validation steps:

```text
1. Update the paper experiment runner to accept explicit ALSS parameters.
2. Run Scenario 3 with the selected ALSS setting.
3. Confirm that the runner output matches the diagnostic behavior.
4. Run a medium validation test, for example 100 trials.
5. Only then run final high-trial experiments.
```

---

## 12. Current Decision

Current decision:

```text
Use ar1/tau=0.25/coreL=3 as the frozen ALSS candidate for the next runner update.
```

Not approved yet:

```text
Final 500-trial or 1000-trial paper run.
Final paper figure regeneration.
Final paper claim update.
```

Approved next step:

```text
Modify the paper experiment runner to accept explicit ALSS mode/tau/coreL parameters.
```

---

## 13. Status

```text
Status: Calibration decision recorded
Selected candidate: ar1, tau=0.25, coreL=3
Final paper run: not yet started
Runner update required: yes
```