# Experiment Summary

## 1. Purpose

This document summarizes the experiment structure and paper-facing validation status of the
`MIMO_GEOMETRY_ANALYSIS` repository.

The goal is to identify:

- which experiments exist,
- which scripts appear to run them,
- which result files and figure files support the paper,
- which trial counts are documented,
- which results are verified,
- which results are paper-facing,
- which older audit notes are retained only as development history.

This document was originally created during the clean rebuild and audit process. The current
Paper 1 result has since been narrowed to a confirmed Z5-focused trial-1000 result.

---

## 2. Current Paper-Facing Status

Current status:

```text
Scenario 3 Z5 trial-1000 result is confirmed and paper-facing.
```

Active paper source:

```text
papers/radarcon2025_alss/ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
```

Archived paper-facing result:

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

Archived paper-facing figure directory:

```text
results/figures/scenario3_trial1000/
```

Figure index:

```text
docs/scenario3_trial1000_figure_index.md
```

Current Paper 1 scope:

```text
Array:              canonical Z5 sparse array
Scenario:           Scenario 3
Estimator:          Coarray MUSIC
Regularization:     ALSS
Trials:             1000 Monte Carlo trials
Coupling cases:     c1 = 0.0 and c1 = 0.3
ALSS mode:          ar1
ALSS tau:           0.25
ALSS coreL:         3
```

Main paper-facing numerical summary:

```text
Reported rows:                         16
Unique operating conditions:           14
Mean improvement over reported rows:   approximately +9.85%
Mean improvement over unique cases:    approximately +9.76%
Worst improvement:                     approximately -0.50%
Best improvement:                      approximately +33.79%
Positive reported rows:                15 / 16
Positive unique conditions:            13 / 14
```

Coupling-level summary:

```text
No coupling, c1 = 0.0:
  Unique conditions:      7
  Mean improvement:       approximately +5.60%
  Positive conditions:    7 / 7

Mutual coupling, c1 = 0.3:
  Unique conditions:      7
  Mean improvement:       approximately +13.92%
  Positive conditions:    6 / 7
```

Important interpretation:

```text
The current paper-facing result is no longer an unresolved trial-count audit.
The Scenario 3 Z5 trial-1000 CSV is the primary quantitative evidence for Paper 1.
Older notes are retained only as historical audit and development context.
```

Conservative claim:

```text
For the canonical Z5 sparse array under Scenario 3, ALSS with ar1/tau=0.25/coreL=3 improves Coarray MUSIC RMSE in most tested conditions, with stronger average gains under mutual coupling than under no-coupling conditions.
```

Limitations:

```text
Paper 1 does not claim universal ALSS optimality.
Paper 1 does not claim full multi-geometry validation.
Paper 1 does not claim physical antenna-pattern correction.
Paper 1 does not include ALSS-II as a validated result.
```

---

## 3. Core Algorithm Files

| Component | File | Paper-Facing Status |
|---|---|---|
| ALSS implementation | `core/radarpy/algorithms/alss.py` | Used by Paper 1; header aligned with trial-1000 claim |
| Coarray MUSIC with ALSS integration | `core/radarpy/algorithms/coarray_music.py` | Used by Paper 1 experiment pipeline |
| Mutual coupling model | `core/radarpy/signal/mutual_coupling.py` | Simulation-based coupling model |
| Snapshot simulation | `core/radarpy/signal/doa_sim_core.py` | Used for Monte Carlo DOA simulations |
| CRB metric | `core/radarpy/algorithms/crb.py` | Available, not the primary Paper 1 evidence |

---

## 4. Paper-Facing Experiment

## 4.1 Scenario 3 — Z5 ALSS Effectiveness

Paper-facing experiment:

```text
Scenario: 3
Array: Z5
Trials: 1000
ALSS mode: ar1
ALSS tau: 0.25
ALSS coreL: 3
Coupling levels: c1=0.0 and c1=0.3
```

Purpose:

```text
Compare baseline Coarray MUSIC and ALSS-enhanced Coarray MUSIC for the canonical Z5 sparse array under no-coupling and mutual-coupling conditions.
```

Archived result:

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

Paper-facing figures:

```text
results/figures/scenario3_trial1000/scenario3_improvement_vs_snr.png
results/figures/scenario3_trial1000/scenario3_rmse_vs_snr_c1_0p3.png
results/figures/scenario3_trial1000/scenario3_improvement_vs_snapshots.png
```

Additional archived figures:

```text
results/figures/scenario3_trial1000/scenario3_rmse_vs_snr_c1_0p0.png
results/figures/scenario3_trial1000/scenario3_harmlessness_vs_snr.png
```

Figure index:

```text
docs/scenario3_trial1000_figure_index.md
```

Status:

```text
Confirmed and paper-facing.
```

---

## 5. Paper-Facing Figure Map

| Paper Figure Candidate | Repository File | Purpose |
|---|---|---|
| Improvement versus SNR | `results/figures/scenario3_trial1000/scenario3_improvement_vs_snr.png` | Shows percentage RMSE improvement for both coupling cases |
| RMSE under mutual coupling | `results/figures/scenario3_trial1000/scenario3_rmse_vs_snr_c1_0p3.png` | Shows baseline versus ALSS RMSE for c1=0.3 |
| Improvement versus snapshots | `results/figures/scenario3_trial1000/scenario3_improvement_vs_snapshots.png` | Shows finite-snapshot robustness trend |
| Z5 sensor geometry | `results/figures/paper1_conceptual/z5_sensor_geometry.png` | Shows canonical Z5 physical sensor positions |
| Z5 coarray weights | `results/figures/paper1_conceptual/z5_coarray_weights.png` | Shows difference-coarray weight distribution |
| Representative MUSIC pseudospectrum | `results/figures/paper1_conceptual/z5_music_pseudospectrum_comparison.png` | Explanatory estimator-level figure, not primary statistical proof |

---

## 6. Reproducibility Requirements for Paper 1

A Paper 1 result is treated as paper-facing only if all of the following are known:

```text
1. script path
2. exact command or documented generation path
3. array geometry
4. SNR range
5. snapshot count
6. number of trials
7. coupling setting
8. ALSS setting
9. output CSV path
10. output figure path
11. paper figure/table reference
```

For the current Paper 1 draft, the primary result satisfying this requirement is:

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

---

## 7. Historical Audit Notes

Earlier development notes mentioned several unresolved issues:

```text
100-trial sweep outputs
500-trial pilot or intermediate outputs
older ALSS settings such as mode=zero, tau=1.0
multi-geometry exploratory figures
ALSS-II exploratory figures
```

These are retained only as development history. They are not the primary Paper 1 evidence.

Paper 1 uses the confirmed Scenario 3 Z5 trial-1000 result with:

```text
mode = ar1
tau = 0.25
coreL = 3
```

---

## 8. Current Risk Assessment

| Risk | Current Status | Action |
|---|---|---|
| Trial-count mismatch | Mitigated for Paper 1 | Use only the archived trial-1000 CSV for paper-facing claims |
| Multiple paper versions | Mitigated | Active paper source is `ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex` |
| Legacy scripts and figures | Still present | Treat as historical or exploratory unless mapped to Paper 1 |
| Generated figures without provenance | Mitigated for Scenario 3 trial-1000 | Use `docs/scenario3_trial1000_figure_index.md` |
| Overclaiming ALSS | Mitigated | Use conservative Z5-focused claim only |
| ALSS-II confusion | Mitigated | ALSS-II is future work, not Paper 1 evidence |

---

## 9. Current Status

```text
Status: Paper 1 Scenario 3 Z5 trial-1000 result confirmed.
Purpose: support final paper cleanup, reproducibility review, and conference submission preparation.
```

---

## 10. Next Actions

1. Compile the active IEEE paper source.
2. Verify that all figures render correctly in the PDF.
3. Check for undefined citations and references.
4. Confirm the conference blind-review policy.
5. Confirm the 6-page conference limit.
6. Prepare the final submission package.
