# Experiment Summary

## 1. Purpose

This document summarizes the current experiment structure of the `MIMO_GEOMETRY_ANALYSIS` repository.

The goal is to identify:

- which experiments exist
- which scripts appear to run them
- which result files or figure files are expected
- which trial counts are documented
- which results are verified
- which results still need audit before paper submission

This document is part of the clean rebuild process.

---

## 2. Current Status

Current status:

```text
Draft / audit stage
```

This repository already contains several experiment-related scripts and documents, but the exact mapping from paper claims to scripts, outputs, and figures is not fully verified yet.

The most important unresolved issue is:

```text
Monte Carlo trial-count consistency
```

Some documents suggest 100 trials per point, some scripts mention 500 trials, and the paper may claim 1000 trials. This must be resolved before final paper submission.

---

## 3. Core Algorithm Files

| Component | File | Status |
|---|---|---|
| ALSS implementation | `core/radarpy/algorithms/alss.py` | Found, needs code review |
| Coarray MUSIC with ALSS integration | `core/radarpy/algorithms/coarray_music.py` | Found, needs code review |
| Mutual coupling model | `core/radarpy/signal/mutual_coupling.py` | Found by import, needs review |
| Snapshot simulation | `core/radarpy/signal/doa_sim_core.py` | Found by import, needs review |
| CRB metric | `core/radarpy/algorithms/crb.py` | Found by import, needs review |

---

## 4. Main Experiment Scripts

## 4.1 Paper Experiment Script

Candidate script:

```text
core/analysis_scripts/run_paper_experiments.py
```

This script appears to define the main paper experiment scenarios.

Documented scenarios:

| Scenario | Name | Purpose | Status |
|---|---|---|---|
| Scenario 1 | Baseline characterization | Compare arrays without coupling and without ALSS | Needs execution |
| Scenario 3 | ALSS effectiveness | Compare baseline vs ALSS under selected conditions | Needs execution |
| Scenario 4 | Cross-array validation | Compare ALSS behavior across arrays | Needs execution |

Documented usage examples include:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario 1 --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario 3 --trials 500 --arrays Z5
python core\analysis_scripts\run_paper_experiments.py --scenario 4 --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 50 --test
```

Important observation:

```text
The script documentation mentions 500 trials for full runs and 50 trials for quick tests.
This does not automatically support a 1000-trial paper claim.
```

Expected output files:

```text
scenario1_baseline.csv
scenario3_alss_effectiveness.csv
scenario4_cross_array.csv
```

Expected output directory:

```text
TBD from script execution
```

Verification status:

```text
Not yet executed during clean rebuild.
```

---

## 4.2 ALSS SNR Sweep Script

Candidate script:

```text
scripts/sweep_alss_snr.ps1
```

Related guide:

```text
papers/radarcon2025_alss/ALSS_SWEEP_GUIDE.md
```

Documented default sweep:

```text
SNR = [-5, 0, 5, 10, 15] dB
Deltas = [2, 13] degrees
Snapshots = 64
Trials = 100 per point
```

Candidate command:

```powershell
.\scripts\sweep_alss_snr.ps1
```

Candidate custom command:

```powershell
.\scripts\sweep_alss_snr.ps1 -SNRs @(-10, 0, 10, 20) -Deltas @(1, 5, 10, 15) -Trials 200
```

Expected output files:

```text
results/alss/baseline_Z5_M64_d2_snr*.csv
results/alss/alss_Z5_M64_d2_snr*.csv
results/alss/baseline_Z5_M64_d13_snr*.csv
results/alss/alss_Z5_M64_d13_snr*.csv
results/alss/all_runs.csv
```

Verification status:

```text
Guide found.
Script execution not yet verified during clean rebuild.
Default trial count appears to be 100 trials per point.
```

---

## 4.3 Plotting Script

Candidate plotting script:

```text
tools/plot_alss_sweep.py
```

Candidate command:

```powershell
python tools\plot_alss_sweep.py results\alss\all_runs.csv
```

Expected generated figures:

```text
results/alss/figures/alss_rmse_delta2.pdf
results/alss/figures/alss_rmse_delta13.pdf
results/alss/figures/alss_resolve_delta2.pdf
results/alss/figures/alss_resolve_delta13.pdf
results/alss/figures/alss_combined_all_deltas.pdf
```

Verification status:

```text
Not yet executed during clean rebuild.
```

---

## 5. Experiment Matrix from Current Evidence

## 5.1 Scenario 1 — Baseline Characterization

Candidate source:

```text
core/analysis_scripts/run_paper_experiments.py
```

Purpose:

```text
Characterize baseline coarray MUSIC behavior without ALSS and without mutual coupling.
```

Likely parameters from script documentation:

| Parameter | Value |
|---|---|
| Arrays | ULA, Nested, Z1, Z4, Z5, Z6 |
| Coupling | 0.0 |
| ALSS | False |
| SNR sweep | -5, 0, 5, 10, 15 dB |
| Snapshot sweep | 32, 64, 128 |
| Metrics | RMSE, CRB ratio, resolution rate, runtime |

Status:

```text
Needs execution and output verification.
```

---

## 5.2 Scenario 3 — ALSS Effectiveness

Candidate source:

```text
core/analysis_scripts/run_paper_experiments.py
```

Purpose:

```text
Compare DOA RMSE with and without ALSS.
```

Likely parameters from script documentation:

| Parameter | Value |
|---|---|
| Focus array | Z5 |
| Coupling | 0.0 and 0.3 |
| ALSS | True / False |
| SNR sweep | -5, 0, 5, 10, 15 dB |
| Snapshot sweep | 32, 64, 128 |
| Metrics | RMSE improvement, p-value, harmlessness, confidence interval |

Status:

```text
Needs execution and output verification.
```

---

## 5.3 Scenario 4 — Cross-Array Validation

Candidate source:

```text
core/analysis_scripts/run_paper_experiments.py
```

Purpose:

```text
Test whether ALSS behavior is consistent across different array geometries.
```

Likely parameters from script documentation:

| Parameter | Value |
|---|---|
| Arrays | ULA, Nested, Z1, Z4, Z5, Z6 |
| Coupling | 0.3 |
| SNR | 5 dB |
| Snapshots | 64 |
| ALSS | True / False |
| Metrics | Relative improvement, ranking consistency, RMSE, resolution rate |

Status:

```text
Needs execution and output verification.
```

---

## 5.4 ALSS SNR Sweep

Candidate source:

```text
scripts/sweep_alss_snr.ps1
```

Purpose:

```text
Run SNR sweeps for baseline and ALSS modes.
```

Likely parameters from guide:

| Parameter | Default |
|---|---|
| Array | Z5 |
| SNR values | -5, 0, 5, 10, 15 dB |
| Delta values | 2, 13 degrees |
| Snapshots | 64 |
| Trials | 100 per point |
| Modes | baseline and ALSS |

Status:

```text
Guide found.
Needs execution and output verification.
```

---

## 6. Known Figure and Result Candidates

Candidate paper figure directory:

```text
papers/radarcon2025_alss/figures/
```

Candidate figure files mentioned during cleanup:

```text
alss_ii_snr_sweep_new.png
nested_snr_sweep.png
z1_snr_sweep.png
z3_2_snr_sweep.png
z4_snr_sweep.png
z5_snr_sweep.png
z6_snr_sweep.png
```

Current status:

```text
These figures exist or were seen as generated/untracked files earlier.
Their generating scripts and trial counts still need verification.
```

Required audit:

| Figure | Script | Trial Count | Verified? |
|---|---|---:|---|
| `z1_snr_sweep.png` | TBD | TBD | No |
| `z3_2_snr_sweep.png` | TBD | TBD | No |
| `z4_snr_sweep.png` | TBD | TBD | No |
| `z5_snr_sweep.png` | TBD | TBD | No |
| `z6_snr_sweep.png` | TBD | TBD | No |
| `nested_snr_sweep.png` | TBD | TBD | No |
| `alss_ii_snr_sweep_new.png` | TBD | TBD | No |

---

## 7. Trial-Count Audit

This is the most important scientific consistency check.

| Source | Trial Count Evidence | Status |
|---|---:|---|
| `papers/radarcon2025_alss/ALSS_SWEEP_GUIDE.md` | 100 trials per point by default | Found, needs confirmation by execution |
| `core/analysis_scripts/run_paper_experiments.py` | 500 trials in documented usage examples | Found, needs confirmation by execution |
| Quick test usage | 50 trials | Found in usage examples |
| Paper manuscript | 1000 trials suspected | Needs direct paper text audit |
| Existing figures | Unknown | Needs mapping |
| Existing CSV outputs | Unknown | Needs inspection |

Decision rule:

```text
If the paper says 1000 trials, the final figure/table must be generated with 1000 trials.
If not, the paper must state the actual trial count.
```

---

## 8. Reproducibility Status Table

| Experiment | Script Found? | Output Found? | Trial Count Known? | Ready for Paper? |
|---|---|---|---|---|
| Scenario 1 baseline | Yes | TBD | 500 in usage examples | No |
| Scenario 3 ALSS effectiveness | Yes | TBD | 500 in usage examples | No |
| Scenario 4 cross-array validation | Yes | TBD | 500 in usage examples | No |
| ALSS SNR sweep | Yes | TBD | 100 default | No |
| Existing paper figures | Partially | Yes/likely | Unknown | No |

---

## Verified Pilot Execution Notes

During the clean rebuild, the paper experiment runner was tested from the repository root using the DS conda environment.

Because the project uses a `src/` package layout, the following PowerShell command is required before directly running the paper experiment script:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$(Get-Location)"

```

Verified pilot commands passed for Scenario 1, Scenario 3, and Scenario 4 using 5 trials with output directories under `$env:TEMP`.
Observed row counts: Scenario 1 = 30 rows, Scenario 3 = 16 rows, Scenario 4 = 12 rows.
These pilot outputs are execution checks only, not paper-ready scientific results, and should not be committed as final results.

---

## 9. Minimum Smoke-Test Plan

Before running expensive experiments, run only small smoke tests.

Suggested smoke-test command:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 5 --test
```

If this fails, do not run 500 or 1000 trials.

After smoke test passes, test individual scenarios:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario 1 --trials 10 --test
python core\analysis_scripts\run_paper_experiments.py --scenario 3 --trials 10 --arrays Z5 --test
python core\analysis_scripts\run_paper_experiments.py --scenario 4 --trials 10 --test
```

Status:

```text
Not yet executed during clean rebuild.
```

---

## 10. Full Experiment Plan

Only after smoke tests pass:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 500
```

For a 1000-trial paper claim, use:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 1000
```

Important:

```text
Do not claim 1000-trial results unless the corresponding output files were generated with --trials 1000.
```

---

## 11. Expected Paper-Ready Requirements

An experiment is paper-ready only if all of the following are true:

```text
1. script path is known
2. exact command is known
3. random seed policy is known
4. array geometry is known
5. SNR range is known
6. snapshot count is known
7. number of trials is known
8. coupling setting is known
9. ALSS setting is known
10. output CSV path is known
11. output figure path is known
12. paper figure/table reference is known
```

---

## 12. Current Risk Assessment

| Risk | Severity | Description |
|---|---|---|
| Trial-count mismatch | High | Paper may claim 1000 trials while some guides/scripts use 100 or 500 |
| Multiple paper versions | High | Active paper source must be confirmed |
| Legacy scripts | Medium | Some scripts may be old or exploratory |
| Generated figures without provenance | High | Existing figures may not be traceable |
| CI not fixed in main | Medium | Does not block documentation, but affects future PR quality |
| Source/result mixing | Medium | Needs later structural cleanup |

---

## 13. Immediate Next Actions

1. Confirm active paper file.
2. Search paper for all mentions of trial count.
3. Search result files for stored trial counts.
4. Run a small smoke test.
5. Record actual output paths.
6. Update this file with verified outputs.
7. Only then prepare paper-ready reproduction commands.

---

## 14. Current Status

```text
Status: Draft
Branch: experiment-summary
Purpose: summarize known experiment structure and identify audit gaps
```
