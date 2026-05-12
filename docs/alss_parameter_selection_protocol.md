# ALSS Parameter-Selection Protocol

## 1. Purpose

This document defines a defensible parameter-selection protocol for Adaptive Lag-Selective Shrinkage (ALSS) before running final high-trial paper experiments.

The goal is to avoid using an arbitrary fixed ALSS configuration such as:

```text
alss_mode = "zero"
alss_tau = 1.0
alss_coreL = 3
```

without evidence that it is stable across SNR, coupling, and array conditions.

This protocol must be finalized before running 500-trial or 1000-trial paper experiments.

---

## 2. Background

Initial pilot experiments showed that ALSS behavior is sensitive to the shrinkage mode and shrinkage strength.

The current paper experiment runner uses a fixed ALSS configuration:

```text
mode = zero
tau = 1.0
coreL = 3
```

Diagnostic sweeps showed that this fixed setting is not consistently optimal.

Some configurations improved with smaller `tau` values or with `ar1` mode, while other configurations degraded when the shrinkage setting was too aggressive.

Therefore, final paper results should not be generated using a fixed ALSS setting unless that setting is selected through a documented protocol.

---

## 3. Parameter Candidates

The ALSS parameter grid should be small, interpretable, and fixed before final evaluation.

Candidate modes:

```text
zero
ar1
```

Candidate shrinkage strengths:

```text
tau = 0.1, 0.25, 0.5, 1.0
```

Core-lag protection:

```text
coreL = 3
```

For the current paper, `coreL` should remain fixed at 3 unless a separate diagnostic study is performed.

Reason:

```text
Changing mode and tau is already sufficient for the first ALSS stability study.
Sweeping coreL at the same time would increase the risk of overfitting.
```

---

## 4. Calibration Versus Final Evaluation

The ALSS parameter-selection process must be separated into two stages.

### Stage 1 — Calibration / Diagnostic Stage

Purpose:

```text
Select a defensible ALSS configuration.
```

Allowed actions:

```text
Run small or medium diagnostic sweeps.
Compare mode/tau settings.
Evaluate robustness across SNR and coupling settings.
Select one frozen ALSS configuration or one clearly defined selection rule.
```

Outputs:

```text
Exploratory diagnostic CSV files.
Summary tables.
Protocol decision.
```

These outputs are not paper-ready final results.

---

### Stage 2 — Final Evaluation Stage

Purpose:

```text
Generate final paper results using the frozen ALSS configuration or frozen selection rule.
```

Allowed actions:

```text
Run 500-trial or 1000-trial experiments.
Regenerate final paper CSV files.
Regenerate final figures and tables.
Report only results produced after the protocol is frozen.
```

Not allowed:

```text
Changing tau/mode after seeing final high-trial results.
Choosing the best ALSS parameter separately for each plotted paper point.
Reporting diagnostic-stage results as final paper results.
```

---

## 5. Recommended Selection Strategy

The recommended strategy is a conservative global-selection protocol.

Instead of choosing a different best parameter for each SNR/coupling condition, select one global ALSS configuration that performs reasonably well across the calibration grid.

Reason:

```text
A global setting is easier to defend scientifically.
It reduces the risk of cherry-picking.
It makes final paper results easier to reproduce.
```

The selected configuration should satisfy:

```text
1. Positive average improvement across calibration conditions.
2. Limited worst-case degradation.
3. Reasonable harmlessness percentage.
4. Stable behavior across coupling and non-coupling cases.
5. No extreme outlier degradation.
```

---

## 6. Calibration Grid

Recommended calibration array:

```text
Z5
```

Reason:

```text
Z5 showed the strongest sensitivity to ALSS tuning and is central to the current paper discussion.
```

Recommended calibration SNR values:

```text
5 dB
10 dB
15 dB
```

Recommended coupling values:

```text
c1 = 0.0
c1 = 0.3
```

Recommended snapshot count:

```text
M = 64
```

Recommended calibration trials:

```text
50 trials minimum
100 trials preferred
```

The 10-trial diagnostic run is useful for debugging only and should not be used to freeze final parameters.

---

## 7. Diagnostic Command

Example 50-trial diagnostic command:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$(Get-Location)"

python tools\diagnose_alss_sweep.py `
    --trials 50 `
    --array Z5 `
    --snrs 5 10 15 `
    --couplings 0.0 0.3 `
    --snapshots 64 `
    --modes zero ar1 `
    --taus 0.1 0.25 0.5 1.0 `
    --output-dir "$env:TEMP\mimo_alss_diag_trial50"
```

Expected output:

```text
$env:TEMP\mimo_alss_diag_trial50\alss_mode_tau_diagnostic.csv
```

The output directory should remain outside the repository unless the result is intentionally archived as a final artifact.

---

## 8. Selection Metrics

The parameter-selection decision should not rely on a single best improvement value.

Use the following metrics:

### 8.1 Mean Improvement

```text
Improvement_% = 100 × (RMSE_baseline - RMSE_ALSS) / RMSE_baseline
```

Positive values indicate improvement.

---

### 8.2 Worst-Case Degradation

For each candidate parameter setting, compute:

```text
Worst_Degradation_% = minimum Improvement_% across calibration conditions
```

A strongly negative value means the setting can severely harm performance.

---

### 8.3 Harmlessness

Harmlessness measures the percentage of trials where ALSS is no worse than baseline.

A candidate setting should not be selected if harmlessness is poor across many conditions.

Recommended threshold:

```text
Average harmlessness >= 70%
```

This threshold is preliminary and can be adjusted if documented.

---

### 8.4 Statistical Significance

Paired p-values should be reported but should not be the only selection criterion during pilot diagnostics.

Reason:

```text
Small calibration runs may not produce statistically significant p-values even when trends are useful.
```

Final statistical claims require high-trial final evaluation.

---

## 9. Recommended Scoring Rule

For each candidate pair:

```text
(mode, tau)
```

compute the following calibration score:

```text
Score = mean(Improvement_%) - 0.5 × abs(min(Improvement_%, 0))
```

Interpretation:

```text
The score rewards average improvement but penalizes harmful worst-case degradation.
```

A candidate should be rejected if:

```text
Worst-case degradation < -20%
```

unless there is a strong documented reason to keep it.

Recommended selection rule:

```text
Choose the candidate with the highest score among candidates whose worst-case degradation is not worse than -20%.
```

If no candidate satisfies this rule, ALSS should be reported as unstable under the current implementation and should not be used for final claims without further algorithm refinement.

---

## 10. Recommended Decision Format

After running the calibration sweep, record the decision in this format:

```text
Selected ALSS mode:
Selected ALSS tau:
Selected ALSS coreL:
Calibration trials:
Calibration SNR values:
Calibration coupling values:
Selection criterion:
Mean improvement:
Worst-case degradation:
Average harmlessness:
Decision date:
```

Example:

```text
Selected ALSS mode: TBD
Selected ALSS tau: TBD
Selected ALSS coreL: 3
Calibration trials: 50 or 100
Calibration SNR values: 5, 10, 15 dB
Calibration coupling values: 0.0, 0.3
Selection criterion: robust score with worst-case degradation constraint
Mean improvement: TBD
Worst-case degradation: TBD
Average harmlessness: TBD
Decision date: TBD
```

---

## 11. Final Paper Execution Rule

Final paper experiments should use only the frozen ALSS setting or frozen selection rule.

Example final command after parameter selection:

```powershell
python core\analysis_scripts\run_paper_experiments.py `
    --scenario all `
    --trials 1000 `
    --output-dir "results\paper_1000_trials"
```

However, the current paper experiment runner may need to be updated before this command can enforce the selected ALSS parameters.

Current known issue:

```text
The paper experiment runner currently calls ALSS with fixed mode/tau values.
```

Therefore, before final paper execution, either:

```text
1. Update the runner to accept ALSS mode/tau command-line arguments, or
2. Create a final experiment runner that uses the frozen selected ALSS parameters.
```

---

## 12. Required Runner Improvement

The final paper runner should expose these command-line arguments:

```text
--alss-mode
--alss-tau
--alss-core-l
```

Recommended defaults before final freeze:

```text
--alss-mode zero
--alss-tau 1.0
--alss-core-l 3
```

But final paper commands should explicitly pass the selected values.

Example:

```powershell
python core\analysis_scripts\run_paper_experiments.py `
    --scenario all `
    --trials 1000 `
    --alss-mode ar1 `
    --alss-tau 0.25 `
    --alss-core-l 3 `
    --output-dir "results\paper_1000_trials"
```

The values above are examples only and must not be treated as final until calibration is complete.

---

## 13. Risk Controls

To reduce overfitting and ensure defensibility:

```text
1. Do not choose the best mode/tau separately for every SNR point.
2. Do not tune on the final 1000-trial output.
3. Keep diagnostic outputs separate from final outputs.
4. Record exact calibration commands.
5. Record exact final evaluation commands.
6. Use fixed random seeds for reproducibility.
7. Do not update paper figures until final results are regenerated.
```

---

## 14. Current Status

```text
Status: Draft protocol
Diagnostic tool: tools/diagnose_alss_sweep.py
Initial diagnostic run: completed with 10 trials
Final ALSS parameters: not yet selected
Final 500/1000-trial paper run: not yet approved
```

---

## 15. Immediate Next Actions

1. Run a 50-trial or 100-trial diagnostic sweep outside the repository.
2. Summarize candidate mode/tau performance.
3. Select one robust ALSS configuration or declare current ALSS unstable.
4. Update the paper experiment runner to accept explicit ALSS parameters.
5. Freeze the selected protocol.
6. Only then run final high-trial paper experiments.

---

## 16. Decision

Current decision:

```text
Do not run 500-trial or 1000-trial final paper experiments yet.
```

Reason:

```text
The ALSS parameter-selection protocol is not frozen.
The current fixed setting mode=zero, tau=1.0 is not sufficiently justified.
```