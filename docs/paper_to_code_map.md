# Paper-to-Code Map

## 1. Purpose

This document maps the ALSS/coarray MUSIC paper content to the repository code, experiment scripts, figures, tables, and verification status.

The goal is to make every major paper claim traceable to:

- source code
- experiment script
- generated result
- figure or table
- verification status

This file is part of the clean rebuild process for making the repository paper-ready and reproducible.

---

## 2. Active Paper Version

Current active paper source:

```text
papers/radarcon2025_alss/ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
```

Current bibliography file:

```text
papers/radarcon2025_alss/references.bib
```

Paper-facing result file:

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

Paper-facing figure directory:

```text
results/figures/scenario3_trial1000/
```

Status:

```text
Scenario 3 Z5 trial-1000 result is confirmed and paper-facing.
```

The current Paper 1 scope is deliberately limited:

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

Main conservative claim:

```text
For the canonical Z5 sparse array under Scenario 3, ALSS with ar1/tau=0.25/coreL=3 improves Coarray MUSIC RMSE in most tested conditions, with stronger average gains under mutual coupling than under no-coupling conditions.
```

Important limitation:

```text
This paper does not claim universal ALSS optimality, full multi-geometry validation, or physical antenna-pattern correction.
```

Historical note:

```text
Earlier audit notes in this repository mentioned unresolved 100-trial, 500-trial, and 1000-trial consistency checks. For the current Paper 1 draft, the paper-facing Scenario 3 Z5 result is the archived 1000-trial CSV listed above.
```

---

## 3. Main Technical Components

| Paper Component | Description | Expected Code Location | Verification Status |
|---|---|---|---|
| Sparse array geometry | Defines physical sensor locations for ULA, Nested, Z1, Z3_2, Z4, Z5, Z6, and other arrays | `geometry_processors/` | Partially available |
| Difference coarray | Computes all pairwise sensor differences and coarray lag weights | `geometry_processors/bases_classes.py` and derived processors | Needs mapping |
| Weight-constrained sparse arrays | Uses array geometries with reduced or zero small-lag weights to reduce mutual coupling sensitivity | `geometry_processors/z*_processor.py` | Needs mapping |
| Mutual coupling model | Applies coupling between closely spaced physical sensors | TBD | Needs identification |
| Coarray MUSIC | Builds virtual covariance / Toeplitz structure and applies MUSIC DOA estimation | `doa_estimation/` or experiment scripts | Needs verification |
| ALSS | Adaptive Lag-Selective Shrinkage applied to coarray lag estimates | TBD | Needs identification |
| Monte Carlo simulation | Repeated trials over SNR, snapshots, coupling, or array geometry | `experiments/`, `analysis_scripts/`, or `tools/` | Needs identification |
| RMSE evaluation | Computes DOA estimation RMSE and performance improvement | `doa_estimation/metrics.py` or scripts | Needs verification |
| Figures and tables | Generates paper plots and numerical summaries | `papers/radarcon2025_alss/figures/`, `results/` | Needs consistency check |

---

## 4. Paper Section Mapping

## 4.1 Introduction

| Paper Claim / Topic | Code / Result Link | Status |
|---|---|---|
| Sparse arrays increase aperture and DOA resolution | Geometry processors and coarray analysis | Needs mapping |
| Mutual coupling degrades DOA performance, especially at small spacings | Mutual coupling experiment scripts | Needs identification |
| Weight-constrained arrays reduce coupling sensitivity | Z-family array processors | Needs verification |
| ALSS improves robustness by reducing high-variance lag effects | ALSS implementation and simulation results | Needs verification |

---

## 4.2 Array Geometry and Difference Coarray

| Item | Expected Implementation | Status |
|---|---|---|
| Physical sensor positions | `geometry_processors/*_processor.py` | Needs review |
| Difference coarray computation | `geometry_processors/bases_classes.py` | Needs review |
| Coarray lag weights | `compute_weight_distribution()` or equivalent | Needs review |
| Contiguous coarray aperture | `analyze_contiguous_segments()` or equivalent | Needs review |
| Coarray holes | `analyze_holes()` or equivalent | Needs review |
| Performance summary table | `generate_performance_summary()` or equivalent | Needs review |

Candidate files:

```text
geometry_processors/bases_classes.py
geometry_processors/ula_processors.py
geometry_processors/nested_processor.py
geometry_processors/z1_processor.py
geometry_processors/z3_1_processor.py
geometry_processors/z3_2_processor.py
geometry_processors/z4_processor.py
geometry_processors/z5_processor.py
geometry_processors/z6_processor.py
```

---

## 4.3 Mutual Coupling Model

| Paper Element | Expected Code | Status |
|---|---|---|
| Coupling matrix definition | TBD | Needs identification |
| Coupling coefficient sweep | TBD | Needs identification |
| Exponential coupling decay model | TBD | Needs verification |
| Coupling applied to received signal / covariance | TBD | Needs verification |

Required action:

```text
Search repository for:
- coupling
- mutual
- mcm
- C_ij
- alpha
- c1
```

---

## 4.4 DOA Estimation and MUSIC

| Paper Element | Expected Code | Status |
|---|---|---|
| Signal model | `doa_estimation/simulation.py` | Needs review |
| Steering vector | `doa_estimation/music.py` or equivalent | Needs review |
| Sample covariance matrix | `doa_estimation/` or experiment scripts | Needs review |
| Coarray covariance / lag estimates | TBD | Needs identification |
| Toeplitz virtual covariance matrix | TBD | Needs identification |
| MUSIC pseudospectrum | `doa_estimation/music.py` | Needs review |
| Peak detection / DOA estimate | `doa_estimation/music.py` | Needs review |

Candidate files:

```text
doa_estimation/music.py
doa_estimation/simulation.py
doa_estimation/metrics.py
doa_estimation/visualization.py
```

---

## 4.5 Adaptive Lag-Selective Shrinkage

| Paper Element | Expected Code | Status |
|---|---|---|
| Per-lag correlation estimate | TBD | Needs identification |
| Lag-dependent shrinkage weight | TBD | Needs identification |
| Core-lag protection | TBD | Needs identification |
| Long-lag variance suppression | TBD | Needs identification |
| ALSS-enhanced Toeplitz matrix | TBD | Needs identification |
| ALSS parameter selection | TBD | Needs identification |

Required action:

```text
Search repository for:
- ALSS
- shrinkage
- lag
- adaptive
- denoise
- coarray denoising
```

---

## 5. Experiment-to-Figure Map

| Figure / Table | Expected Script | Output File | Trials | Status |
|---|---|---|---|---|
| Array geometry comparison | TBD | TBD | N/A | Needs mapping |
| Z1 SNR sweep | TBD | `z1_snr_sweep.png` | TBD | Needs verification |
| Z3_2 SNR sweep | TBD | `z3_2_snr_sweep.png` | TBD | Needs verification |
| Z4 SNR sweep | TBD | `z4_snr_sweep.png` | TBD | Needs verification |
| Z5 SNR sweep | TBD | `z5_snr_sweep.png` | TBD | Needs verification |
| Z6 SNR sweep | TBD | `z6_snr_sweep.png` | TBD | Needs verification |
| Nested SNR sweep | TBD | `nested_snr_sweep.png` | TBD | Needs verification |
| ALSS-II SNR sweep | TBD | `alss_ii_snr_sweep_new.png` | TBD | Needs verification |

Known figure directory:

```text
papers/radarcon2025_alss/figures/
```

---

## 6. Result Consistency Checklist

Before the paper is finalized, every result must answer these questions:

- Which script generated this result?
- Which commit generated this result?
- What random seed was used?
- How many Monte Carlo trials were used?
- What SNR range was used?
- What number of snapshots was used?
- Which array geometry was used?
- Was mutual coupling enabled?
- Was ALSS enabled?
- Where is the output figure/table stored?
- Does the paper claim match the generated result?

---

## 7. Monte Carlo Trial Count Audit

This is currently a high-priority issue.

| Result | Paper Claim | Actual Trial Count | Status |
|---|---:|---:|---|
| Z1 SNR sweep | 1000 | TBD | Needs audit |
| Z3_2 SNR sweep | 1000 | TBD | Needs audit |
| Z5 SNR sweep | 1000 | TBD | Needs audit |
| Nested SNR sweep | 1000 | TBD | Needs audit |
| ALSS-II sweep | 1000 | TBD | Needs audit |

Decision rule:

```text
If the figure was generated with 100 trials, either regenerate with 1000 trials or update the paper text.
```

---

## 8. Reproducibility Requirements

A result is considered reproducible only if the repository contains:

- the script used to generate it
- the required input parameters
- the random seed policy
- the number of trials
- the output file path
- a short explanation of expected runtime
- a saved figure or table, if appropriate

---

## 9. Future FPGA/HLS Mapping

The paper includes a possible future hardware implementation path.

Potential software-to-hardware mapping:

| Algorithm Step | FPGA/HLS Candidate? | Notes |
|---|---|---|
| Sample covariance matrix formation | Yes | Already connected to FPGA/HLS SCM accelerator work |
| Coarray lag accumulation | Yes | Suitable for parallel accumulation |
| Toeplitz matrix construction | Yes | Structured memory/data movement |
| MUSIC spectrum scan | Yes | Parallel angle-grid evaluation |
| Jacobi/EVD preprocessing | Possible | More complex but relevant |
| ALSS shrinkage computation | Yes | Per-lag weighting is lightweight |
| Peak detection | Yes | Can be implemented as streaming comparison |

Connection to related repository:

```text
https://github.com/hmolhem/fpga-hls-doa-scm-accelerator
```

---

## 10. Immediate Action Items

1. Identify active paper source file.
2. Identify all scripts that generate paper figures.
3. Identify all scripts that generate paper tables.
4. Verify 1000-trial versus 100-trial results.
5. Create `docs/reproduction_guide.md`.
6. Create `docs/experiment_summary.md`.
7. Update README after the mapping is verified.
8. Defer source restructuring until the paper-to-code map is complete.

---

## 11. Current Status

```text
Status: Draft
Branch: paper-map
Next review target: identify active scripts and result files
```