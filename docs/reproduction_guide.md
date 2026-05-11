# Reproduction Guide

## 1. Purpose

This document explains how to reproduce, audit, and validate the ALSS/coarray MUSIC experiments in this repository.

The immediate goal is not to claim that all paper results are already final. The immediate goal is to identify which scripts generate which results, which trial counts were used, and which outputs can support the paper.

This guide is part of the clean rebuild process.

---

## 2. Current Reproduction Status

Current status:

```text
Draft / audit stage
```

The repository contains useful experiment code and documentation, but the result chain is not fully verified yet.

The most important unresolved issue is the Monte Carlo trial-count consistency:

```text
Some documents or paper claims mention 1000 trials.
Some sweep guides and scripts appear to use 100 or 500 trials.
This must be audited before final paper claims are accepted.
```

---

## 3. Environment

Recommended local environment used during the rebuild:

```text
Conda environment: DS
Python version: 3.12.12
```

Useful development tools:

```text
ruff
pytest
numpy
scipy
pandas
matplotlib
```

Basic environment check:

```powershell
python --version
where.exe python
python -m pytest --version
python -m ruff --version
```

Expected Python path when the DS environment is active:

```text
C:\Users\hosse\miniforge3\envs\DS\python.exe
```

---

## 4. Repository Files Relevant to Reproduction

### 4.1 ALSS Implementation

Candidate implementation file:

```text
core/radarpy/algorithms/alss.py
```

Important function:

```text
apply_alss(...)
```

This function applies Adaptive Lag-Selective Shrinkage to lag-domain coarray estimates.

Expected concepts implemented here:

- per-lag shrinkage
- coarray lag weights
- core-lag protection
- zero or AR(1) shrinkage target
- Hermitian symmetry enforcement

---

### 4.2 Coarray MUSIC Implementation

Candidate implementation file:

```text
core/radarpy/algorithms/coarray_music.py
```

Important function:

```text
estimate_doa_coarray_music(...)
```

Expected functions or concepts:

- virtual ULA covariance construction
- forward-backward averaging
- diagonal loading
- MUSIC pseudospectrum
- peak picking
- optional ALSS integration

---

### 4.3 Main Paper Experiment Script

Candidate paper experiment script:

```text
core/analysis_scripts/run_paper_experiments.py
```

This script appears to define major experimental scenarios:

```text
Scenario 1: Baseline characterization
Scenario 3: ALSS effectiveness
Scenario 4: Cross-array validation
```

Important note:

```text
The script documentation mentions usage with --trials 500.
This is not the same as 1000 trials.
```

---

### 4.4 ALSS Sweep Guide

Candidate sweep guide:

```text
papers/radarcon2025_alss/ALSS_SWEEP_GUIDE.md
```

Important note:

```text
The sweep guide describes default SNR sweeps using 100 trials per point.
This is a major audit item if the paper claims 1000 trials.
```

---

### 4.5 Active Paper Candidate

Candidate paper file:

```text
papers/ALSS_IEEE_Paper_Complete.tex
```

Status:

```text
Needs verification
```

Required action:

```text
Confirm whether this is the active paper version or whether another paper file should be treated as the current manuscript.
```

---

## 5. Basic Local Validation Commands

Before running reproduction experiments, verify the repository state.

```powershell
git checkout main
git pull origin main
git status
```

Expected clean status:

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

For the current documentation branch:

```powershell
git checkout reproduction-guide
git status
```

Expected branch status:

```text
On branch reproduction-guide
nothing to commit, working tree clean
```


## Smoke-Test Notes

During the clean rebuild, the paper experiment runner was tested locally from the repository root.

Because the project uses a `src/` package layout, the Python path must include both the repository root and the `src` directory before running the paper experiment script directly.

PowerShell setup command:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$(Get-Location)"
---

## 6. Test Commands

At the time of the rebuild, the following local checks were known to pass on the DS conda environment when linting was scoped to tests:

```powershell
python -m ruff check tests --ignore E402,E741
python -m pytest -q
```

Known local result from previous CI cleanup work:

```text
Ruff: all checks passed
Pytest: 6 passed
```

Important:

```text
The main branch CI is not fully fixed yet.
The ci-cleanup branch contains proposed CI/lint changes but has not been merged into main.
```

---

## 7. Candidate Experiment Commands

### 7.1 Paper Experiment Script

Candidate command pattern:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 500
```

Candidate individual scenarios:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario 1 --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario 3 --trials 500 --arrays Z5
python core\analysis_scripts\run_paper_experiments.py --scenario 4 --trials 500
```

Quick test pattern:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 50 --test
```

Status:

```text
Not yet verified in this clean rebuild.
These commands must be tested before being treated as official reproduction commands.
```

---

### 7.2 ALSS SNR Sweep Script

Candidate sweep script:

```text
scripts/sweep_alss_snr.ps1
```

Candidate default command:

```powershell
.\scripts\sweep_alss_snr.ps1
```

Important default behavior from the guide:

```text
SNR = [-5, 0, 5, 10, 15] dB
Deltas = [2, 13] degrees
Snapshots = 64
Trials = 100 per point
```

Custom trial-count example:

```powershell
.\scripts\sweep_alss_snr.ps1 -SNRs @(-10, 0, 10, 20) -Deltas @(1, 5, 10, 15) -Trials 200
```

Status:

```text
Needs execution and verification.
```

---

## 8. Expected Output Locations

Candidate output locations mentioned in existing documentation:

```text
results/alss/
results/alss/all_runs.csv
results/alss/figures/
```

Candidate output files from the sweep guide:

```text
results/alss/baseline_Z5_M64_d2_snr*.csv
results/alss/alss_Z5_M64_d2_snr*.csv
results/alss/baseline_Z5_M64_d13_snr*.csv
results/alss/alss_Z5_M64_d13_snr*.csv
results/alss/all_runs.csv
```

Candidate plotting command:

```powershell
python tools\plot_alss_sweep.py results\alss\all_runs.csv
```

Candidate generated figures:

```text
results/alss/figures/alss_rmse_delta2.pdf
results/alss/figures/alss_rmse_delta13.pdf
results/alss/figures/alss_resolve_delta2.pdf
results/alss/figures/alss_resolve_delta13.pdf
results/alss/figures/alss_combined_all_deltas.pdf
```

Status:

```text
Needs verification against actual generated files and paper figures.
```

---

## 9. Monte Carlo Trial-Count Audit

This audit is mandatory.

| Experiment / Figure | Claimed Trials | Script / Guide Evidence | Verified? | Action |
|---|---:|---:|---|---|
| Default ALSS SNR sweep | TBD | 100 trials/point in sweep guide | No | Verify |
| Paper experiment script | TBD | 500 trials in script usage examples | No | Verify |
| Paper manuscript claims | 1000 suspected | Needs paper review | No | Verify |
| Final paper figures | TBD | Unknown | No | Map to script/output |

Decision rule:

```text
If the paper says 1000 trials, the figure/table must be generated with 1000 trials.
If not, the paper must clearly state the actual trial count.
```

---

## 10. Reproducibility Checklist

A result is reproducible only if the following are known:

```text
1. script path
2. command line used
3. random seed policy
4. array geometry
5. SNR range
6. snapshot count
7. number of trials
8. coupling setting
9. ALSS on/off
10. output CSV path
11. output figure/table path
12. matching paper figure/table
```

---

## 11. Paper-to-Code Traceability Table

| Paper Element | Candidate Code / Script | Status |
|---|---|---|
| ALSS method | `core/radarpy/algorithms/alss.py` | Found, needs review |
| Coarray MUSIC | `core/radarpy/algorithms/coarray_music.py` | Found, needs review |
| Paper experiments | `core/analysis_scripts/run_paper_experiments.py` | Found, needs execution |
| ALSS SNR sweep | `scripts/sweep_alss_snr.ps1` | Found, needs execution |
| Sweep plotting | `tools/plot_alss_sweep.py` | Found, needs execution |
| Active paper | `papers/ALSS_IEEE_Paper_Complete.tex` | Candidate, needs confirmation |
| Trial-count audit | paper + scripts + outputs | Not complete |

---

## 12. Current Known Risks

1. Trial count inconsistency between paper claims and sweep scripts.
2. Multiple paper versions may exist.
3. Some results may be generated from legacy scripts.
4. CI is not fully fixed in main.
5. Generated files and source files may still be mixed.
6. The exact script-to-figure mapping is not yet complete.

---

## 13. Immediate Next Actions

The next concrete actions are:

1. Confirm the active paper file.
2. Search paper text for trial-count claims.
3. Search scripts for default trial counts.
4. Identify which scripts generate the known paper figures.
5. Run a small smoke-test experiment, not a full 1000-trial run.
6. Record actual output file paths.
7. Update this guide with verified commands only.

---

## 14. Current Status

```text
Status: Draft
Branch: reproduction-guide
Purpose: document verified reproduction workflow and identify unresolved audit items
```