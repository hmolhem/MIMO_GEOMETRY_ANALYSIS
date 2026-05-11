# Paper Audit Report

## 1. Purpose

This document audits the current ALSS paper files, experiment scripts, and trial-count claims in the `MIMO_GEOMETRY_ANALYSIS` repository.

The goal is to identify whether the paper claims, scripts, figures, and reported numerical results are scientifically consistent.

This audit is part of the clean rebuild process.

---

## 2. Current Audit Status

```text
Status: Draft / evidence collection stage
Branch: paper-audit
```

This audit does not modify algorithms, experiment scripts, or paper results.

Its purpose is to document:

- active paper candidates
- trial-count claims
- script-level trial-count evidence
- result-generation uncertainty
- paper submission risks
- required corrections before final submission

---

## 3. Main Finding

The current evidence shows a serious trial-count consistency issue.

The paper text includes claims of:

```text
1000 Monte Carlo trials / runs
```

However, comments inside the paper and existing guide/script evidence indicate that some numerical results and figures may still come from earlier lower-trial runs, especially:

```text
100 trials
500 trials
50-trial quick tests
```

Therefore, the current paper should not be treated as final until the trial-count issue is resolved.

---

## 4. Active Paper Candidates

The repository contains multiple paper versions.

Candidate paper files found during the audit include:

```text
papers/ALSS_IEEE_Paper_Complete.tex
papers/ALSS_IEEE_Paper_Complete01.tex
papers/radarcon2025_alss/ALSS_COMBINED_IEEE_Paper.tex
papers/radarcon2025_alss/ALSS_COMBIined_IEEE_PAPER_NEW.tex
papers/radarcon2025_alss/ALSS_II_IEEE_Paper.tex
papers/radarcon2025_alss/alss_paper_section.tex
```

Current active paper candidate:

```text
papers/ALSS_IEEE_Paper_Complete.tex
```

Reason:

```text
This file appears to be the main complete IEEE-style paper currently referenced by the clean rebuild documentation.
```

Status:

```text
Needs confirmation before final submission.
```

Required action:

```text
Decide which paper file is the official active manuscript.
Archive or clearly label older paper drafts.
```

---

## 5. Paper-Level Trial-Count Evidence

The paper search found trial-related lines in:

```text
papers/ALSS_IEEE_Paper_Complete.tex
papers/ALSS_IEEE_Paper_Complete01.tex
papers/radarcon2025_alss/ALSS_COMBINED_IEEE_Paper.tex
papers/radarcon2025_alss/ALSS_COMBIined_IEEE_PAPER_NEW.tex
papers/radarcon2025_alss/ALSS_II_IEEE_Paper.tex
papers/radarcon2025_alss/alss_paper_section.tex
```

Important observation from the active paper candidate:

```text
The paper claims 1000 Monte Carlo runs per condition.
```

However, internal comments in the paper indicate that some numerical values or figures may still reflect earlier 100-trial runs.

Example issue type:

```text
The manuscript may say 1000 trials, while comments indicate that table or figure values still need regeneration from 1000-trial runs.
```

This is a high-priority academic consistency issue.

---

## 6. Script-Level Trial-Count Evidence

The script search found trial-related evidence in many files, including:

```text
core/analysis_scripts/run_paper_experiments.py
core/analysis_scripts/run_scenario1_baseline.py
core/analysis_scripts/run_scenario2_coupling_impact.py
core/analysis_scripts/run_scenario3_alss_regularization.py
core/analysis_scripts/run_scenario4_array_comparison.py
core/analysis_scripts/run_scenario5_coupling_models.py
analysis_scripts/generate_paper_data.py
analysis_scripts/generate_paper_data_final.py
analysis_scripts/analyze_alss_mcm_enhanced.py
analysis_scripts/analyze_alss_mcm_scenarios.py
scripts/sweep_alss_snr.ps1
tools/plot_alss_sweep.py
tools/plot_paper_benchmarks.py
```

The main candidate experiment script is:

```text
core/analysis_scripts/run_paper_experiments.py
```

Known documented usage patterns include:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario 1 --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario 3 --trials 500 --arrays Z5
python core\analysis_scripts\run_paper_experiments.py --scenario 4 --trials 500
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 50 --test
```

Therefore, `run_paper_experiments.py` appears to support configurable trial counts, but the documented full-run examples use 500 trials, not 1000.

---

## 7. ALSS Sweep Trial-Count Evidence

The ALSS sweep guide points to:

```text
scripts/sweep_alss_snr.ps1
```

The documented default sweep appears to use:

```text
100 trials per point
```

Typical sweep configuration:

```text
SNR = [-5, 0, 5, 10, 15] dB
Deltas = [2, 13] degrees
Snapshots = 64
Trials = 100 per point
```

This is inconsistent with a manuscript-level claim of 1000 trials unless the final paper figures were regenerated separately using 1000 trials.

Required action:

```text
Verify whether paper figures came from default 100-trial sweeps or separate 1000-trial runs.
```

---

## 8. Figure and Result Provenance Risk

The current paper includes figures and tables that likely depend on previously generated outputs.

Candidate figure/result names discussed during cleanup include:

```text
z1_snr_sweep.png
z3_2_snr_sweep.png
z4_snr_sweep.png
z5_snr_sweep.png
z6_snr_sweep.png
nested_snr_sweep.png
alss_ii_snr_sweep_new.png
alss_mcm_gap_reduction.png
alss_mcm_bias_variance_decomposition.png
```

Current status:

```text
The generating script and exact trial count for each figure are not fully verified.
```

This is a high-risk issue because a figure cannot support a 1000-trial paper claim unless its generating command and result file are known.

---

## 9. Trial-Count Audit Table

| Source / Artifact | Trial Count Evidence | Status | Risk |
|---|---:|---|---|
| `papers/ALSS_IEEE_Paper_Complete.tex` | Claims 1000 trials in experimental framework and captions | Found | High |
| Paper internal comments | Indicate some values/figures may still reflect 100-trial data | Found | High |
| `core/analysis_scripts/run_paper_experiments.py` | Usage examples with 500 trials and 50-trial test mode | Found | Medium |
| `scripts/sweep_alss_snr.ps1` / sweep guide | Default appears to be 100 trials per point | Found | High |
| Existing figures | Trial count unknown | Not verified | High |
| Existing CSV outputs | Trial count unknown unless stored in file | Not verified | High |
| Reference WCSA-style validation | Often expects large Monte Carlo validation | Relevant | Medium |

---

## 10. Required Decision Before Submission

Before paper submission, choose one of the following paths.

### Option A — Regenerate Final Results with 1000 Trials

Use this option if the paper will keep the claim:

```text
1000 Monte Carlo trials
```

Required work:

```text
1. Run final experiments with --trials 1000.
2. Save output CSV files in a clearly named results directory.
3. Regenerate all paper figures from those CSV files.
4. Update all tables using 1000-trial values.
5. Remove manuscript comments saying values still come from 100-trial runs.
6. Record exact commands in docs/reproduction_guide.md.
7. Record output-to-figure mapping in docs/paper_to_code_map.md.
```

Recommended output directory naming:

```text
results/paper_1000_trials/
```

Recommended figure directory naming:

```text
papers/radarcon2025_alss/figures/final_1000_trials/
```

---

### Option B — Change Paper Claims to Actual Trial Counts

Use this option if there is not enough time to regenerate all results with 1000 trials.

Required work:

```text
1. Determine actual trial count for each figure/table.
2. Change paper text from 1000 trials to the actual value.
3. Make captions specific, e.g., 100 trials/point or 500 trials.
4. Avoid claiming stronger statistical significance than the data supports.
5. Document the limitation clearly.
```

This option is faster, but weaker academically.

---

## 11. Recommended Path

Recommended path:

```text
Use Option A for final paper-quality results.
```

Reason:

```text
The paper is intended to be technically defensible and portfolio-quality.
A 1000-trial validation is more consistent with the current manuscript wording and stronger for academic review.
```

However, before running expensive 1000-trial experiments, run smoke tests.

---

## 12. Smoke-Test Plan

Before full regeneration, run small tests.

Recommended commands:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 5 --test
```

If that works, test individual scenarios:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario 1 --trials 10 --test
python core\analysis_scripts\run_paper_experiments.py --scenario 3 --trials 10 --arrays Z5 --test
python core\analysis_scripts\run_paper_experiments.py --scenario 4 --trials 10 --test
```

Only after these pass should larger runs be attempted.

---

## 13. Candidate Full-Run Commands

If the active script is confirmed to work, candidate final commands are:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario all --trials 1000
```

Or scenario-by-scenario:

```powershell
python core\analysis_scripts\run_paper_experiments.py --scenario 1 --trials 1000
python core\analysis_scripts\run_paper_experiments.py --scenario 3 --trials 1000 --arrays Z5
python core\analysis_scripts\run_paper_experiments.py --scenario 4 --trials 1000
```

Status:

```text
Not yet executed during paper-audit branch.
```

Important:

```text
These commands are candidate commands only.
They must be tested before being declared official reproduction commands.
```

---

## 14. Active Paper Cleanup Required

The active paper file must be cleaned before final submission.

Required paper cleanup tasks:

```text
1. Remove or resolve all comments saying results still come from 100-trial runs.
2. Ensure every table caption has the correct trial count.
3. Ensure every figure caption has the correct trial count.
4. Ensure experimental framework section matches actual scripts.
5. Ensure random seed claims match actual script behavior.
6. Ensure SNR, snapshot count, coupling settings, and array geometry match outputs.
7. Ensure all figures exist at the paths used by LaTeX.
```

---

## 15. Paper Version Control Required

The repository contains multiple paper versions.

Recommended cleanup plan:

```text
1. Keep only one active paper manuscript in the primary paper path.
2. Move old drafts to an archive folder.
3. Add a README explaining which file is active.
4. Do not delete old drafts until the active manuscript is verified.
```

Candidate structure:

```text
papers/
├── ALSS_IEEE_Paper_Complete.tex        # active manuscript
└── archive/
    ├── ALSS_IEEE_Paper_Complete01.tex
    └── radarcon2025_old_drafts/
```

Status:

```text
Not yet performed.
```

---

## 16. Minimum Requirements for a Defensible Result

A result is defensible only if the following are known:

```text
1. active paper file
2. script path
3. exact command
4. random seed policy
5. array geometry
6. SNR values
7. snapshot count
8. coupling parameters
9. ALSS on/off configuration
10. number of Monte Carlo trials
11. output CSV file
12. generated figure file
13. table or figure reference in paper
```

---

## 17. Current Conclusion

The project is technically promising, but the current paper is not yet submission-safe.

Main reason:

```text
The paper currently contains 1000-trial claims while some comments and guide/script evidence suggest that parts of the results may still come from 100-trial or 500-trial workflows.
```

Therefore:

```text
Do not submit the paper until the trial-count audit is resolved.
Do not claim 1000-trial results unless the corresponding outputs were actually generated with 1000 trials.
```

---

## 18. Immediate Next Actions

1. Confirm `papers/ALSS_IEEE_Paper_Complete.tex` as the active manuscript or choose another active file.
2. Extract all paper lines containing `1000`, `trial`, `Monte Carlo`, and `seed`.
3. Extract all script defaults for `--trials`.
4. Run smoke tests with 5–10 trials.
5. Record output file locations.
6. Decide between 1000-trial regeneration or paper text correction.
7. Update `docs/reproduction_guide.md`, `docs/experiment_summary.md`, and `docs/paper_to_code_map.md` after verification.

---

## 19. Branch Status

```text
Branch: paper-audit
Purpose: audit paper claims, trial counts, and result consistency
No code changes intended
```