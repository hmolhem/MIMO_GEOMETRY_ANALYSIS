# Getting Started Guide

## Installation

### Prerequisites

Before installing, ensure you have:

- **Windows 10/11** with PowerShell 5.1+
- **Python 3.13.0** (managed via pyenv)
- **Git** for cloning the repository

### Step 1: Clone Repository

```powershell
cd C:\MyDocument\RadarPy
git clone <repository-url> MIMO_GEOMETRY_ANALYSIS
cd MIMO_GEOMETRY_ANALYSIS
```

### Step 2: Activate Virtual Environment

The project includes a pre-configured virtual environment with all dependencies:

```powershell
# Using PowerShell (recommended)
.\mimo-geom-dev\Scripts\Activate.ps1

# Using Command Prompt
.\mimo-geom-dev\Scripts\activate.bat

# Verify activation (prompt should show "(mimo-geom-dev)")
python --version  # Should display: Python 3.13.0
```

### Step 3: Verify Dependencies

```powershell
pip list | Select-String "numpy|pandas|matplotlib|scipy"
```

**Expected Output:**
```
matplotlib     3.10.7
numpy          2.3.4
pandas         2.3.3
scipy          1.15.0
```

If dependencies are missing, install them:

```powershell
pip install -r requirements.txt
```

---

## Your First Analysis

### Interactive Mode (Recommended for Beginners)

Launch the graphical demo for menu-driven exploration:

```powershell
python analysis_scripts/graphical_demo.py
```

**Menu Options:**
1. ULA (Uniform Linear Array)
2. Nested Array
3. Z1-Z6 Specialized Arrays

**What happens:**
- Interactive prompts for array parameters (N, d)
- Automatic 7-step coarray analysis
- ASCII visualization of virtual array
- Performance metrics table printed to terminal

### Command-Line Mode

For scripted analysis with output options:

```powershell
# Z5 array with 7 sensors, save CSV summary
python analysis_scripts/run_z5_demo.py --N 7 --d 1.0 --markdown --save-csv

# ULA with 4 sensors, JSON metadata
python analysis_scripts/run_ula_demo.py --M 4 --save-json

# Z4 array with assertions (validate theoretical properties)
python analysis_scripts/run_z4_demo_.py --N 7 --assert --show-weights
```

**Output Locations:**
- **Terminal**: Markdown-formatted tables with `--markdown`
- **CSV**: `results/summaries/{array}_summary_N{N}_d{d}.csv`
- **JSON**: `results/summaries/{array}_metadata_N{N}_d{d}.json`

### Understanding Output

**Sample Z5 Output (N=7):**

```markdown
| Metric                        | Value  |
|------------------------------|--------|
| Physical Sensors (N)          | 7      |
| Virtual Sensors (Mv)          | 43     |
| Coarray Aperture              | 42     |
| Contiguous Segment Length (L) | 43     |
| Max Detectable Sources (K_max)| 21     |
| Weight at Lag 1               | 8      |
| Holes in [-21, 21]            | 0      |
```

**Key Metrics Explained:**
- **Mv**: Number of unique virtual sensor positions
- **K_max**: Maximum number of sources the array can estimate
- **Weight at Lag 1**: Frequency of lag-1 differences (higher is better for accuracy)
- **Holes**: Missing positions in the virtual array

---

## Running DOA Benchmarks

### Basic Benchmark

```powershell
python core/analysis_scripts/run_benchmarks.py `
  --arrays Z5 `
  --N 7 `
  --algs CoarrayMUSIC `
  --snr 0,5,10 `
  --snapshots 64 `
  --k 2 `
  --delta 13 `
  --trials 100 `
  --out results/bench/test.csv
```

**Parameters:**
- `--arrays`: Array types (ULA, Z4, Z5, Z6, Nested)
- `--N`: Number of physical sensors
- `--snr`: Signal-to-noise ratios in dB (comma-separated)
- `--snapshots`: Number of temporal snapshots
- `--k`: Number of sources (2 for two-source DOA)
- `--delta`: Angular separation between sources (degrees)
- `--trials`: Monte Carlo repetitions for statistical averaging

### With ALSS Regularization

```powershell
python core/analysis_scripts/run_benchmarks.py `
  --arrays Z5 `
  --N 7 `
  --algs CoarrayMUSIC `
  --snr 5 `
  --snapshots 64 `
  --k 2 `
  --delta 13 `
  --trials 200 `
  --alss on `
  --alss-mode zero `
  --alss-tau 1.0 `
  --alss-coreL 3 `
  --out results/bench/alss_test.csv
```

**ALSS Parameters:**
- `--alss on`: Enable Adaptive Lag-Selective Shrinkage
- `--alss-mode zero`: Shrink toward zero (alternative: `mean`)
- `--alss-tau 1.0`: Shrinkage intensity (0=no shrinkage, 1=full shrinkage)
- `--alss-coreL 3`: Number of protected lags (no shrinkage applied)

### Expected Runtime

- **100 trials**: ~2-3 minutes
- **400 trials**: ~10-15 minutes
- **Full paper benchmark (25 scenarios × 400 trials)**: ~4-6 hours

---

## Visualizing Results

### Generate Plots from CSV

```powershell
python tools/plot_paper_benchmarks.py `
  results/bench/z5_paper_N7_T400_alss_zero.csv `
  --all
```

**Generated Plots (in `results/bench/figures/`):**
1. `rmse_with_ci_*.png` - RMSE with bootstrap confidence intervals
2. `resolve_with_ci_*.png` - Resolve rate with Wilson intervals
3. `condition_numbers_*.png` - Rx/Rv condition numbers
4. `combined_comparison_*.png` - 2×2 grid for hard cases (Δ ≤ 20°)

**Output Formats:**
- PNG (300 DPI, publication-ready)
- PDF (vector graphics for LaTeX)

### Viewing Plots

```powershell
# Open figures folder in Explorer
explorer results\bench\figures
```

---

## Troubleshooting

### Issue: "Python version mismatch"

**Error:** `python --version` shows Python 3.12 or 3.11 instead of 3.13.0

**Solution:**
```powershell
# Ensure pyenv is configured for Python 3.13.0
pyenv versions
pyenv local 3.13.0

# Recreate virtual environment if needed
python -m venv mimo-geom-dev --clear
.\mimo-geom-dev\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Module not found: geometry_processors"

**Error:** `ModuleNotFoundError: No module named 'geometry_processors'`

**Solution:**
```powershell
# Ensure you're in the project root directory
cd C:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS

# Analysis scripts use relative imports
python analysis_scripts\graphical_demo.py  # ✅ Correct
python graphical_demo.py                   # ❌ Incorrect (wrong working directory)
```

### Issue: "Virtual environment not activating"

**Error:** PowerShell execution policy blocks script activation

**Solution:**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user (one-time setup)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Try activation again
.\mimo-geom-dev\Scripts\Activate.ps1
```

### Issue: "Benchmark takes too long"

**Problem:** 400-trial benchmarks run for hours

**Solution:**
```powershell
# Start with small trial count for testing
python scripts/run_paper_benchmarks.py --array Z5 --trials 50

# Use fewer scenarios
python scripts/run_paper_benchmarks.py `
  --array Z5 `
  --trials 400 `
  --deltas 13,20 `      # Only 2 deltas instead of 5
  --snr-vals 5,10,15    # Only 3 SNR values instead of 5
```

### Issue: "SVD files not generated"

**Problem:** `--dump-svd` flag doesn't create CSV files

**Solution:**
```powershell
# Ensure results/svd/ directory exists
mkdir -Force results\svd

# Verify flag syntax
python core/analysis_scripts/run_benchmarks.py `
  --arrays Z5 `
  --N 7 `
  --dump-svd `  # ✅ Correct position (before --out)
  --out results/bench/test.csv
```

---

## Next Steps

1. **Compare Arrays**: See [TUTORIAL_01_ARRAY_COMPARISON.md](tutorials/TUTORIAL_01_ARRAY_COMPARISON.md)
2. **DOA Estimation**: See [TUTORIAL_02_DOA_ESTIMATION.md](tutorials/TUTORIAL_02_DOA_ESTIMATION.md)
3. **ALSS Regularization**: See [TUTORIAL_03_ALSS_REGULARIZATION.md](tutorials/TUTORIAL_03_ALSS_REGULARIZATION.md)
4. **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md) for programmatic usage

---

**Questions?** Check the [FAQ](FAQ.md) or open an issue on GitHub.
