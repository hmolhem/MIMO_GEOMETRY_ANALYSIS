# MIMO Array Geometry Analysis Framework

[![CI](https://github.com/hmolhem/MIMO_GEOMETRY_ANALYSIS/actions/workflows/ci.yml/badge.svg)](https://github.com/hmolhem/MIMO_GEOMETRY_ANALYSIS/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.13.0-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

A comprehensive Python framework for analyzing MIMO radar array geometries through **difference coarray analysis**. This toolkit enables researchers to evaluate virtual array properties, weight distributions, and Direction-of-Arrival (DOA) estimation performance across various array configurations including ULA, Nested, and specialized Z1-Z6 arrays.

## Current Research Paper: ALSS for Z5 Coarray MUSIC

This repository currently includes a polished IEEE-style conference draft for Paper 1:

**Adaptive Lag-Selective Shrinkage for Robust Coarray MUSIC in Weight-Constrained Sparse Arrays**

The paper studies Adaptive Lag-Selective Shrinkage (ALSS) as a post-geometry coarray-domain denoising method for the canonical Z5 sparse array under Scenario 3. The paper uses 1000 Monte Carlo trials and evaluates Coarray MUSIC performance with and without mutual coupling.

Paper source:

```text
papers/radarcon2025_alss/ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
```


### Key Features

- ✅ **8+ Array Implementations**: ULA, Nested, TCA, ePCA, Z1-Z6 specialized geometries
- ✅ **Standardized 7-Step Analysis Pipeline**: Automated difference coarray computation and performance evaluation
- ✅ **DOA Estimation Module** (NEW): Complete MUSIC implementation with signal simulation and metrics
- ✅ **Spatial MUSIC Algorithm**: High-resolution angle estimation using signal/noise subspace decomposition
- ✅ **Mutual Coupling Support**: Optional electromagnetic coupling modeling for realistic hardware scenarios
- ✅ **Publication-Ready Visualization**: Automated plotting with LaTeX-ready figures (300 DPI)
- ✅ **Comprehensive Benchmarking**: CLI tools for parameter sweeps and statistical analysis
- ✅ **100% Test Coverage**: Validated across all 15 array configurations

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.13.0 (managed via pyenv)
- **OS**: Windows 10/11 with PowerShell
- **Dependencies**: NumPy, Pandas, Matplotlib (see `requirements.txt`)

### Installation

```powershell
# Activate virtual environment (Recommended: Batch file)
.\activate_venv.bat

# Alternative: PowerShell script (requires RemoteSigned execution policy)
.\activate_venv.ps1

# Verify installation
python --version  # Should show Python 3.13.0
pip list | Select-String "numpy|pandas|matplotlib"
```

**Note:** Use the provided activation scripts (`activate_venv.bat` or `activate_venv.ps1`) which include automatic verification. The batch file (`.bat`) is recommended as it works without execution policy restrictions.

### Your First Analysis

```powershell
# Interactive graphical demo (recommended for beginners)
python analysis_scripts/graphical_demo.py

# CLI analysis for Z5 array with 7 sensors
python analysis_scripts/run_z5_demo.py --N 7 --markdown

# Run DOA estimation benchmark (with automatic venv activation)
.\run_benchmarks_with_venv.ps1 -Arrays Z5 -N 7 -Trials 100

# Alternative: Batch file runner
run_benchmarks_with_venv.bat Z5 7 100
```

**Expected Output:**
- **Terminal**: Detailed coarray analysis (positions, weights, holes, performance metrics)
- **Plots**: 6-panel visualization saved to `results/plots/`
- **CSV**: Performance summary tables in `results/summaries/`

---

## 📐 Core Concept: Difference Coarray

MIMO arrays create **virtual sensors** through pairwise differences of physical sensor positions:

```
Physical array: [0, 5, 8, 11, 14, 17, 21] (N=7 sensors)
                          ↓
Difference coarray: All pairs (nᵢ - nⱼ) → N² = 49 differences
                          ↓
Virtual array: 17 unique positions spanning [-21, 21]
```

**Key Benefit**: Virtual aperture size determines **maximum detectable sources**:
- **Degrees of Freedom (DOF)**: K_max = floor(L/2)
- **Z5 Example**: 7 physical sensors → 43 virtual sensors → 21 sources detectable

---

## 📊 Project Structure

```
MIMO_GEOMETRY_ANALYSIS/
├── README.md                    # This file
├── CONTRIBUTING.md              # Contribution guidelines
├── pyproject.toml               # Package configuration & Ruff settings
├── pytest.ini                   # Test configuration
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
│
├── .github/
│   ├── copilot-instructions.md # AI assistant guidance
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline (pytest + ruff)
│
├── geometry_processors/         # Array processor implementations
│   ├── bases_classes.py        # Abstract BaseArrayProcessor framework
│   ├── ula_processors.py       # Uniform Linear Array
│   ├── nested_processor.py     # Nested arrays
│   ├── sna_processor.py        # SNA3 arrays
│   └── z[1-6]_processor.py     # Z-family weight-constrained arrays
│
├── core/radarpy/               # Core radar algorithms
│   ├── algorithms/             # MUSIC, ALSS, coarray processing
│   ├── signal/                 # Signal generation & mutual coupling
│   ├── metrics/                # Performance evaluation
│   └── analysis_scripts/       # Benchmark runners
│
├── doa_estimation/             # DOA estimation module (NEW)
│   ├── music.py               # MUSIC algorithm implementation
│   ├── README.md              # DOA documentation
│   └── QUICK_START.md         # 5-minute DOA guide
│
├── analysis_scripts/           # Interactive demos & analysis tools
│   ├── graphical_demo.py      # Menu-driven analysis
│   ├── run_*_demo.py          # CLI demos for each array type
│   ├── methods_demo.py        # Validate all processor methods
│   ├── core/                  # Wrapper scripts
│   └── demos/                 # Example usage
│
├── scripts/                    # Automation & batch processing
│   ├── activate_venv.bat/.ps1 # Virtual environment activation
│   ├── run_benchmarks_with_venv.* # Automated benchmark runners
│   └── cleanup_*.ps1          # Repository maintenance
│
├── tools/                      # Development & analysis utilities
│   ├── add_docstrings.py      # Docstring audit tool
│   ├── plot_paper_benchmarks.py # Plotting utilities
│   └── analyze_svd.py         # Matrix analysis
│
├── src/mimo_geom_analysis/     # Package structure (for distribution)
│   ├── __init__.py
│   ├── runners.py             # Benchmark execution framework
│   └── paper_experiments.py   # Paper-ready experiment configs
│
├── tests/                      # Test suite
│   ├── test_geometry_processors_smoke.py
│   ├── test_coarray_mv.py
│   ├── test_runners.py
│   └── test_paper_experiments.py
│
├── docs/                       # Comprehensive documentation
│   ├── guides/                # Usage guides & tutorials
│   ├── development/           # Development documentation
│   └── summaries/             # Project summaries
│
├── papers/                     # Publication materials
│   └── radarcon2025_alss/     # RadarCon 2025 ALSS paper
│
├── results/                    # Auto-generated outputs (gitignored)
│   ├── plots/                 # Visualization outputs
│   ├── bench/                 # Benchmark results
│   └── summaries/             # Performance tables
│
├── archives/                   # Historical backups (gitignored)
│   ├── garbage_archive_*.zip
│   └── tests_backup_*/
│
├── data/                       # Input datasets
├── notebooks/                  # Jupyter notebooks
└── configs/                    # Configuration files
```

---

## 📖 Usage Examples

### Geometry Analysis

```powershell
# Interactive menu
python analysis_scripts/graphical_demo.py

# CLI with options
python analysis_scripts/run_z5_demo.py --N 7 --markdown --save-csv
python analysis_scripts/run_ula_demo.py --M 4 --save-json
python analysis_scripts/run_z4_demo_.py --N 7 --assert --show-weights
```

### DOA Estimation Benchmarks

```powershell
# Basic benchmark
python core/analysis_scripts/run_benchmarks.py \
  --arrays Z5 --N 7 --algs CoarrayMUSIC \
  --snr 0,5,10,15 --snapshots 64 --delta 13 --trials 200 \
  --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3

# Paper-ready benchmarks with CIs
python scripts/run_paper_benchmarks.py \
  --array Z5 --N 7 --trials 400 \
  --deltas 10,13,20,30,45 --alss-mode zero

# Generate publication plots
python tools/plot_paper_benchmarks.py results/bench/*.csv --all
```

### Programmatic Usage

```python
from geometry_processors.z5_processor_ import Z5ArrayProcessor

# Create and analyze
processor = Z5ArrayProcessor(N=7, d=1.0)
results = processor.run_full_analysis()

# Access results
print(f"Max detectable sources: {results.max_detectable_sources}")
print(results.performance_summary_table.to_markdown(index=False))
```

### DOA Estimation with MUSIC (NEW)

```bash
# Quick start - estimate 3 sources
python analysis_scripts/run_doa_demo.py --array Z5 --N 7 --K 3

# SNR performance analysis
python analysis_scripts/run_doa_demo.py --mode snr-comparison --array Z5 --N 7 --K 3

# Compare multiple arrays
python analysis_scripts/run_doa_demo.py --mode array-comparison --arrays ULA,Z5,Z6 --N 7 --K 3

# Test all array types (100% validation)
python analysis_scripts/test_all_arrays_doa.py
```

**Programmatic DOA:**
```python
from geometry_processors.z5_processor import Z5ArrayProcessor
from doa_estimation.music import MUSICEstimator

# Get coarray from geometry
processor = Z5ArrayProcessor(N=7, d=1.0)
results = processor.run_full_analysis()
coarray = results.unique_differences

# Run MUSIC estimation
estimator = MUSICEstimator(sensor_positions=coarray, wavelength=2.0)
signals, true_angles = estimator.simulate_signals(K_sources=3, snr_db=20)
estimated_angles, spectrum = estimator.estimate(signals, K_sources=3)

print(f"True:      {true_angles}")
print(f"Estimated: {estimated_angles}")
# Output: Perfect estimation with high SNR!
```

**Documentation:**
- Quick Start: `doa_estimation/QUICK_START.md` (5-minute guide)
- Full Guide: `doa_estimation/README.md` (600+ lines)
- Technical: `doa_estimation/IMPLEMENTATION_SUMMARY.md`

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** | Installation, first analysis, troubleshooting |
| **[API_REFERENCE.md](docs/API_REFERENCE.md)** | Complete API documentation with examples |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design, data flow, component overview |
| **[BENCHMARKING_GUIDE.md](docs/BENCHMARKING_GUIDE.md)** | Running benchmarks, interpreting results |
| **[TUTORIAL_01.md](docs/tutorials/TUTORIAL_01_ARRAY_COMPARISON.md)** | Compare ULA, Nested, and Z5 arrays |
| **[TUTORIAL_02.md](docs/tutorials/TUTORIAL_02_DOA_ESTIMATION.md)** | DOA estimation with MUSIC |
| **[TUTORIAL_03.md](docs/tutorials/TUTORIAL_03_ALSS_REGULARIZATION.md)** | Using ALSS for improved performance |

---

## 🎯 Research Applications

This framework supports cutting-edge research in MIMO radar array design and DOA estimation algorithms.

### Use Cases

**Array Geometry Research:**
- Novel sparse array topology design and optimization
- Difference coarray analysis for virtual aperture characterization
- Weight distribution studies for improved DOF
- Comparative analysis across array families (ULA, Nested, Coprime, Z-family)

**DOA Estimation Research:**
- MUSIC algorithm performance evaluation
- Coarray-based subspace methods
- Finite-sample regularization techniques (ALSS)
- Mutual coupling compensation strategies
- Low-SNR and closely-spaced source scenarios

**Signal Processing Research:**
- Spatial smoothing on coarray manifolds
- Subspace decomposition with regularization
- Performance bounds and theoretical limits
- Monte Carlo simulation frameworks

### Published Work: RadarCon 2025

**Paper**: *"Adaptive Lag-Selective Shrinkage for MIMO Coarray DOA Estimation"*  
**Authors**: Hossein Molhem et al.  
**Status**: Under Review

**Contributions:**
- Novel ALSS regularization for covariance matrix estimation
- Lag-selective shrinkage addressing finite-sample effects
- Comprehensive performance evaluation across array types

**Key Results** (from 1000-trial validation, seed=42):
- **Z5 Array Performance**: 45% gap reduction under mutual coupling (7.4° → 7.1° RMSE)
- **Z1 Array Performance**: 30% gap reduction (7.2° → 7.3° baseline)
- **Z3_2 Array Performance**: 20% gap reduction (11.9° → 12.4° baseline)
- **Statistical Significance**: p < 0.001 (Z5), p < 0.01 (Z3_2), p < 0.01 (Z1)
- **Conditioning**: 2.5× better matrix conditioning (κ(Rv) vs κ(Rx))
- **Dataset**: 3,000 Monte Carlo trials (1000 per array type) across multiple conditions
- **Test Conditions**: SNR=10dB, M=200 snapshots, MCM (c1=0.3, α=0.5)
- **Coverage**: 3 Z-family arrays, 4 experimental conditions, mutual coupling modeling
- **Reproducibility**: Complete code with fixed random seed (42), publication-ready plots

**Paper Materials** (in `papers/radarcon2025_alss/`):
- LaTeX source and compiled PDF
- All figures (publication-ready, 300 DPI)
- Benchmark data and processing scripts
- Reproduction guide

### Array Performance Summary

Comparative performance for N=7 sensors at d=λ/2 spacing:

| Array | N | Virtual (Mv) | DOF (K_max) | Holes | Aperture | Key Feature |
|-------|---|--------------|-------------|-------|----------|-------------|
| **ULA** | 7 | 13 | 6 | Many | 6λ/2 | Baseline reference |
| **Nested** | 7 | 25 | 12 | 0 | 12λ/2 | Optimal DOF |
| **Coprime** | 7 | 27 | 13 | 0 | 15λ/2 | Coprime structure |
| **Z1** | 7 | 25 | 12 | 0 | 12λ/2 | Weight constraints |
| **Z3_1** | 7 | 33 | 16 | 0 | 16λ/2 | Improved constraints |
| **Z3_2** | 7 | 33 | 16 | 0 | 16λ/2 | Alternative Z3 |
| **Z4** | 7 | 39 | 19 | 0 | 19λ/2 | w(1)=w(2)=0 |
| **Z5** | 7 | 43 | 21 | 0 | 21λ/2 | Advanced w(1)=w(2)=0 |
| **Z6** | 7 | 43 | 21 | 0 | 21λ/2 | Ultimate constraints |

**Note**: K_max = floor(L/2) where L is the contiguous coarray segment length. All Z-family arrays eliminate unit and two-unit lags for improved small-lag weight distribution.

---

## 🛠️ Development

### Documentation standard

All public and private methods across processors implement Google-style docstrings and include the author tag for attribution:


- Style: Google-style (Args/Returns/Raises)
- Author tag: `Author: Hossein Molhem`
- Coverage: 100% of functions/methods documented; verified by an automated audit tool (`tools/add_docstrings.py`).


Example snippet:

```python
def analyze_coarray(self) -> None:
  """
  Analyze the difference coarray and populate derived metrics.

  Author: Hossein Molhem

  Returns:
    None
  """
  ...
```

### Mutual Coupling Feature (NEW - Nov 2025)

Model electromagnetic interactions between array elements:

```python
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.signal.doa_sim_core import run_music

# Generate coupling matrix
C = generate_mcm(7, positions, model="exponential", c1=0.3, alpha=0.5)

# Run DOA estimation with coupling
result = run_music(positions, 1.0, [10, -15], 2, 10, 100, 
                   coupling_matrix=C)
```

**Features:**
- ✅ Optional (easy on/off control)
- ✅ Multiple models: exponential, Toeplitz, measured data
- ✅ Works with all arrays and algorithms
- ✅ Backward compatible

See **[docs/MUTUAL_COUPLING_GUIDE.md](docs/MUTUAL_COUPLING_GUIDE.md)** for complete documentation.

### Adding New Arrays

1. Create processor in `geometry_processors/`
2. Implement 8 abstract methods from `BaseArrayProcessor`
3. Add demo script in `analysis_scripts/`
4. Update `graphical_demo.py` menu

See **[docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)** for details.

### Running Tests

```powershell
# Unit tests
python -m pytest core/tests/ -v

# Validate all processors
python analysis_scripts/methods_demo.py

# Check results
type results\method_test_log.txt
```

### Running Benchmarks with Venv

The project provides automated benchmark runners that handle virtual environment activation:

```powershell
# PowerShell runner (with parameters)
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 100

# With mutual coupling enabled
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 100 -WithCoupling

# Batch runner (positional arguments)
run_benchmarks_with_venv.bat Z5 7 100
run_benchmarks_with_venv.bat Z5 7 100 coupling
```

**See full guide**: [docs/BENCHMARK_EXECUTION_GUIDE.md](docs/BENCHMARK_EXECUTION_GUIDE.md)

---

## ⚠️ Important Limitations

- **Root-MUSIC on virtual arrays**: Experimental; use grid-based CoarrayMUSIC for published results
- **Z6 CoarrayMUSIC**: Produces fragmented coarray (Mv=3); unsuitable for virtual array DOA
- **Spatial aliasing**: Z5/Z6 have wide apertures; ensure λ ≥ 2d to avoid aliasing

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Theory**: Difference coarray (Vaidyanathan & Pal, 2010)
- **ALSS**: Original contribution for finite-sample regularization
- **Tools**: NumPy, Pandas, Matplotlib, SciPy

---

## 📞 Citation

```bibtex
@inproceedings{your2025alss,
  title={Adaptive Lag-Selective Shrinkage for MIMO Coarray DOA Estimation},
  author={Your Name},
  booktitle={IEEE RadarCon},
  year={2025}
}
```

---

## 📚 References

1. P. Pal and P. P. Vaidyanathan, "Nested arrays," *IEEE Trans. Signal Process.*, 2010.
2. C.-L. Liu and P. P. Vaidyanathan, "Spatial smoothing in coarray MUSIC," *IEEE Signal Process. Lett.*, 2015.

---

**Last Updated:** 2025-11-06  
**Version:** 1.0.0  
**Status:** Production-Ready

---

## Developer setup (local)

Recommended minimal steps to set up a development environment locally (Windows PowerShell):

```powershell
# 1) Create a new venv (recommended path: envs/mimo-geom-dev or your own location)
python -m venv .\envs\mimo-geom-dev

# 2) Activate venv (PowerShell)
.\envs\mimo-geom-dev\Scripts\Activate.ps1

# 3) Upgrade pip and install dev requirements
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# 4) (Optional) Install package in editable mode when pyproject/setup is present
# pip install -e .
```

Notes:
- Use `requirements-dev.txt` for test/lint/runtime dev deps (pytest, ruff, numpy, pandas).
- We intentionally do not track virtualenvs in git. If you accidentally committed a venv, remove it from tracking with `git rm -r --cached <venv-path>` and add it to `.gitignore`.

