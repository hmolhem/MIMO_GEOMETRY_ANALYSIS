# MIMO Geometry Analysis Framework

[![CI](https://github.com/hmolhem/MIMO_GEOMETRY_ANALYSIS/actions/workflows/ci.yml/badge.svg)](https://github.com/hmolhem/MIMO_GEOMETRY_ANALYSIS/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository is a research-grade Python framework for sparse-array direction-of-arrival (DOA) estimation, difference-coarray analysis, mutual-coupling-aware simulation, and Adaptive Lag-Selective Shrinkage (ALSS) regularization.

The current research focus is:

```text
Weight-constrained sparse arrays
        +
finite-snapshot coarray lag regularization
        +
Coarray MUSIC
        +
future FPGA/HLS acceleration path
```

The project is intended to support:

- reproducible DOA experiments,
- IEEE-style paper development,
- sparse-array geometry analysis,
- ALSS method validation,
- GitHub portfolio presentation for DSP / FPGA / radar signal-processing roles.

---

## Current Paper 1: ALSS for Z5 Coarray MUSIC

This repository currently includes a polished IEEE-style conference draft for Paper 1:

**Adaptive Lag-Selective Shrinkage for Robust Coarray MUSIC in Weight-Constrained Sparse Arrays**

Paper source:

```text
papers/radarcon2025_alss/ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
```

References:

```text
papers/radarcon2025_alss/references.bib
```

### Paper 1 Scope

Paper 1 is deliberately focused:

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

This paper does **not** claim that ALSS is universally optimal for all sparse arrays. It establishes a compact, reproducible, Z5-focused conference result.

### Main Scientific Idea

Weight-constrained sparse arrays reduce mutual-coupling sensitivity through geometry. For example, the Z5 geometry suppresses critical small-lag coarray weights such as:

```text
w(1) = 0
w(2) = 0
```

However, the same sparse coarray-weight distribution can produce unequal finite-snapshot estimation variance across lags. ALSS is introduced as a post-geometry coarray-domain regularization step:

```text
X -> Rhat_x -> rhat[l] -> rhat_ALSS[l] -> Rhat_v -> Coarray MUSIC
```

Important distinction:

- Z5 geometry helps reduce coupling-related sensitivity.
- ALSS helps reduce finite-snapshot lag-estimation variance.
- ALSS does not modify the physical antenna pattern.
- ALSS does not modify the physical sensor geometry.
- ALSS does not explicitly estimate or invert the mutual-coupling matrix.

---

## Paper 1 Main Result

The Scenario 3 Z5 trial-1000 result shows:

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

Conservative paper claim:

```text
For the canonical Z5 sparse array under Scenario 3, ALSS with ar1/tau=0.25/coreL=3 improves Coarray MUSIC RMSE in most tested conditions, with stronger average gains under mutual coupling than under no-coupling conditions.
```

Avoid overclaiming:

```text
Do not claim ALSS always improves every trial.
Do not claim universal ALSS optimality across all arrays.
Do not claim ALSS physically changes the antenna pattern.
Do not claim full multi-geometry validation from the Z5-only result.
```

---

## Key Paper 1 Files

### Main Paper

```text
papers/radarcon2025_alss/ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
papers/radarcon2025_alss/references.bib
```

### Trial-1000 Data

```text
results/paper_experiments/scenario3_z5_ar1_tau025_trial1000.csv
```

### Trial-1000 Figures

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

### Conceptual Figures

```text
results/figures/paper1_conceptual/z5_sensor_geometry.png
results/figures/paper1_conceptual/z5_coarray_weights.png
results/figures/paper1_conceptual/z5_music_pseudospectrum_comparison.png
results/figures/paper1_conceptual/README.md
```

### Figure and Result Documentation

```text
docs/scenario3_trial1000_figure_index.md
docs/paper1_conceptual_figure_plan.md
docs/paper_to_code_map.md
docs/experiment_summary.md
```

### Figure Generation Tools

```text
tools/plot_paper1_z5_conceptual_figures.py
tools/plot_paper1_z5_music_pseudospectrum.py
tools/plot_scenario3_results.py
```

---

## Reproducing Paper 1 Figures

### Generate Z5 Conceptual Figures

```powershell
python tools\plot_paper1_z5_conceptual_figures.py
```

Expected conceptual outputs:

```text
results/figures/paper1_conceptual/z5_sensor_geometry.png
results/figures/paper1_conceptual/z5_coarray_weights.png
```

The canonical Z5 positions used for Paper 1 are:

```text
[0, 5, 8, 11, 14, 17, 21]
```

The expected small-lag weights reported by the script are:

```text
w(1) = 0
w(2) = 0
w(3) = 4
w(4) = 1
w(5) = 1
```

### Generate Representative MUSIC Pseudospectrum Figure

```powershell
python tools\plot_paper1_z5_music_pseudospectrum.py
```

Expected output:

```text
results/figures/paper1_conceptual/z5_music_pseudospectrum_comparison.png
```

Representative pseudospectrum case:

```text
True DOAs:      [-20.0, 15.0]
SNR:            15 dB
Snapshots:      64
Coupling:       c1 = 0.3
Selected seed:  0
ALSS mode:      ar1
ALSS tau:       0.25
ALSS coreL:     3
```

Script-reported estimates:

```text
No coupling / baseline:  [-20.0, 15.0]
Coupled / baseline:      [-20.0, 14.9]
Coupled / ALSS:          [-20.0, 15.0]
```

Scientific role:

```text
This figure is explanatory.
It is not the primary statistical proof.
The trial-1000 RMSE and improvement plots remain the primary quantitative evidence.
```

### Generate Scenario 3 Trial-1000 Figures

```powershell
python tools\plot_scenario3_results.py
```

Expected output directory:

```text
results/figures/scenario3_trial1000/
```

Paper-facing figures:

```text
scenario3_improvement_vs_snr.png
scenario3_rmse_vs_snr_c1_0p3.png
scenario3_improvement_vs_snapshots.png
```

---

## Compiling the IEEE Paper

Use PowerShell from the repository root.

```powershell
$tmp = "$env:TEMP\alss_trial1000_compile_readme_check"
New-Item -ItemType Directory -Force $tmp
Copy-Item papers\radarcon2025_alss\references.bib $tmp\references.bib

Push-Location papers\radarcon2025_alss

pdflatex `
    -interaction=nonstopmode `
    -output-directory $tmp `
    ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex

Pop-Location

Push-Location $tmp
bibtex ALSS_SCENARIO3_Z5_TRIAL1000_IEEE
Pop-Location

Push-Location papers\radarcon2025_alss

pdflatex `
    -interaction=nonstopmode `
    -output-directory $tmp `
    ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex

pdflatex `
    -interaction=nonstopmode `
    -output-directory $tmp `
    ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex

Pop-Location
```

Check the log:

```powershell
Select-String -Path "$env:TEMP\alss_trial1000_compile_readme_check\ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.log" -Pattern "Output written|Fatal error|Emergency stop|Citation.*undefined|Reference.*undefined|undefined references|not found|LaTeX Warning"
```

Open the PDF:

```powershell
Start-Process msedge "$env:TEMP\alss_trial1000_compile_readme_check\ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.pdf"
```

---

## Project Structure

```text
MIMO_GEOMETRY_ANALYSIS/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── pytest.ini
│
├── geometry_processors/
│   ├── bases_classes.py
│   ├── ula_processors.py
│   ├── nested_processor.py
│   ├── sna_processor.py
│   └── z*_processor.py
│
├── core/
│   ├── radarpy/
│   │   ├── algorithms/
│   │   │   ├── alss.py
│   │   │   ├── coarray.py
│   │   │   └── coarray_music.py
│   │   ├── signal/
│   │   │   ├── doa_sim_core.py
│   │   │   └── mutual_coupling.py
│   │   └── metrics/
│   │
│   └── analysis_scripts/
│
├── tools/
│   ├── plot_paper1_z5_conceptual_figures.py
│   ├── plot_paper1_z5_music_pseudospectrum.py
│   └── plot_scenario3_results.py
│
├── docs/
│   ├── scenario3_trial1000_figure_index.md
│   ├── paper1_conceptual_figure_plan.md
│   ├── paper_to_code_map.md
│   ├── experiment_summary.md
│   └── Technical_Proposal_Roadmap.tex
│
├── papers/
│   └── radarcon2025_alss/
│       ├── ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
│       └── references.bib
│
├── results/
│   ├── paper_experiments/
│   └── figures/
│       ├── scenario3_trial1000/
│       └── paper1_conceptual/
│
├── tests/
├── scripts/
├── notebooks/
├── data/
└── configs/
```

---

## Setup

### Prerequisites

Recommended local environment:

```text
Conda environment: DS
Python version: 3.12.x
Tested locally with: Python 3.12.12
Operating system: Windows / PowerShell
```

Core scientific packages:

```text
NumPy
Pandas
Matplotlib
SciPy
```

Development and quality-check tools:

```text
pytest
ruff
```

`pytest` is used to run the project test suite.  
`ruff` is used for code-style and lint checks.

The environment name `DS` is the local development environment used for this project. Other users may choose a different Conda environment name, but Python 3.12.x is recommended.

Create and activate the Conda environment:

```powershell
conda create -n DS python=3.12
conda activate DS
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Verify the environment:

```powershell
python --version
python -m pytest --version
python -m ruff --version
```

---

## Running Tests

Run the available test suite:

```powershell
python -m pytest -v
```

Run Ruff checks if configured:

```powershell
python -m ruff check .
```

This repository is under active research development. Test coverage and validation should be interpreted from the current CI and test outputs, not from a fixed coverage claim.

---

## Example Usage

### Run Z5 Geometry Demo

```powershell
python analysis_scripts\run_z5_demo.py --N 7 --markdown
```

### Run DOA Benchmark

```powershell
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 `
  --N 7 `
  --algs CoarrayMUSIC `
  --snr 0,5,10,15 `
  --snapshots 64 `
  --trials 200 `
  --alss on `
  --alss-mode ar1 `
  --alss-tau 0.25 `
  --alss-coreL 3
```

### Generate Paper-Facing Figures

```powershell
python tools\plot_scenario3_results.py
python tools\plot_paper1_z5_conceptual_figures.py
python tools\plot_paper1_z5_music_pseudospectrum.py
```

---

## Research Positioning

The project follows this research chain:

```text
Weight-constrained sparse-array geometry
        ↓
difference-coarray lag estimation
        ↓
finite-snapshot variance imbalance
        ↓
ALSS coarray-domain regularization
        ↓
virtual covariance reconstruction
        ↓
Coarray MUSIC
        ↓
future FPGA/HLS acceleration
```

The near-term research objective is to finalize Paper 1 as a conservative, reproducible Z5-focused IEEE-style conference paper.

The longer-term roadmap includes:

1. validation across additional geometries such as Z1, Z3, Z4, and Z6,
2. ALSS parameter sensitivity studies,
3. ALSS-II development after independent validation,
4. FPGA/HLS acceleration of selected kernels:
   - sample covariance matrix formation,
   - coarray lag averaging,
   - virtual Toeplitz covariance reconstruction,
   - MUSIC pseudospectrum scanning.

---

## Relationship to FPGA/HLS Work

This repository connects naturally to FPGA acceleration because the DOA processing chain contains hardware-friendly kernels.

Candidate acceleration path:

```text
Sample covariance matrix formation
        ↓
coarray lag averaging
        ↓
ALSS lag regularization
        ↓
Toeplitz covariance reconstruction
        ↓
MUSIC spectrum scan
```

The existing SCM/HLS project provides a foundation for the first stage of this pipeline. Future work should evaluate fixed-point arithmetic, memory partitioning, pipelining, and PS/PL integration for a deployable embedded DOA pipeline.

---

## Limitations

Current Paper 1 limitations:

- Paper 1 is Z5-focused.
- The representative MUSIC pseudospectrum is explanatory, not statistical proof.
- The mutual-coupling model is simulation-based and not a full electromagnetic antenna model.
- Only one fixed ALSS configuration is used in the primary trial-1000 result.
- Broader multi-geometry validation is future work.
- ALSS-II is not part of the current Paper 1 claim.

Repository limitations:

- Some older scripts and documentation may remain from earlier development phases.
- Result paths and APIs may evolve as the repository is cleaned for publication.
- Claims should be tied to committed scripts, archived CSVs, generated figures, and paper-to-code documentation.

---

## Citation

A formal citation will be added after submission or publication.

Working citation placeholder:

```bibtex
@misc{molhem2026alss_z5,
  title  = {Adaptive Lag-Selective Shrinkage for Robust Coarray MUSIC in Weight-Constrained Sparse Arrays},
  author = {Hossein Molhem},
  year   = {2026},
  note   = {Research draft, MIMO_GEOMETRY_ANALYSIS repository}
}
```

---

## License

This project is released under the MIT License. See:

```text
LICENSE
```

---

## Acknowledgments

This research builds on foundational work in:

- MUSIC and subspace-based DOA estimation,
- nested and coprime sparse arrays,
- difference-coarray processing,
- mutual-coupling-aware sparse-array design,
- covariance shrinkage and regularization,
- FPGA/HLS acceleration for radar signal processing.

The current Paper 1 specifically builds on the idea that weight-constrained sparse-array geometry can reduce mutual-coupling sensitivity, while ALSS adds a complementary finite-snapshot statistical denoising layer.

---

## Current Status

```text
Status: active research repository
Current focus: Paper 1 ALSS/Z5 IEEE-style conference draft
Primary result: Scenario 3 Z5 trial-1000 validation
Next step: README cleanup, final paper visual review, and reproducibility check
```

Last updated: May 2026
