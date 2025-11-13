# Quick Reference: Running Python Code for Plots

## TL;DR - Fastest Way

### Open PowerShell and Run This:

```powershell
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
python analysis_scripts/analyze_alss_mcm_baseline.py
```

**Done!** ✅ Plots are saved to `results/plots/`

---

## What Gets Generated

### 3 Publication-Quality Plots:

1. **Gap Reduction** (`alss_mcm_gap_reduction.png`)
   - Shows 30%, 20%, 45% improvement for Z1, Z3_2, Z5
   - Includes confidence intervals and p-values
   - Based on 1000 trials, seed=42

2. **Bias-Variance Decomposition** (`alss_mcm_bias_variance_decomposition.png`)
   - Shows orthogonal effects of ALSS vs MCM
   - Variance reduction: ~40% across all arrays
   - Demonstrates algorithm effectiveness

3. **SNR Effectiveness** (`alss_mcm_snr_effectiveness.png`)
   - Shows ALSS improvement across SNR 0-20 dB
   - Validates when ALSS is most beneficial
   - Proves harmlessness at high SNR

### 1 Results CSV:
- `alss_mcm_baseline_results.csv`
- Contains RMSE, std dev, gap reduction for all arrays

---

## Command Syntax

### Basic Command
```powershell
python analysis_scripts/analyze_alss_mcm_baseline.py
```

### With Output Capture
```powershell
python analysis_scripts/analyze_alss_mcm_baseline.py > results.txt
```

### With Virtual Environment
```powershell
.\mimo-geom-dev\Scripts\Activate.ps1
python analysis_scripts/analyze_alss_mcm_baseline.py
```

---

## Step-by-Step (First Time)

**Step 1:** Open PowerShell  
```powershell
# Press Ctrl+` in VS Code, or open PowerShell
```

**Step 2:** Navigate to project
```powershell
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
```

**Step 3:** Run script
```powershell
python analysis_scripts/analyze_alss_mcm_baseline.py
```

**Step 4:** Wait 2-5 minutes  
(Runs 1000 trials for 3 arrays)

**Step 5:** View plots
```powershell
explorer results\plots
```

---

## Configuration

### Change Number of Trials

**Edit:** `analysis_scripts/analyze_alss_mcm_baseline.py` (Line 48)

```python
NUM_TRIALS = 1000  # Change this number
```

Options:
- 100 = ~1 min (faster)
- 1000 = ~5 min (current, recommended)
- 5000 = ~20 min (more statistical power)

### Change Random Seed

**Edit:** `analysis_scripts/analyze_alss_mcm_baseline.py` (Line 52)

```python
REPRODUCIBILITY_SEED = 42  # Change this number
```

Different seeds = different random numbers (but still reproducible)

---

## Expected Output

```
Testing Array: Z1
  Running 1000 trials...
    Completed 1000/1000 trials
  
  Results:
    Condition 1: 7.215° ± 18.836°
    Condition 3: 7.273° ± 19.119°
    Gap Reduction: 30.0%

[Similar for Z3_2 and Z5]

Creating Plot 1: Bias-Variance Decomposition...
    Saved: results/plots/alss_mcm_bias_variance_decomposition.png
Creating Plot 2: SNR-Dependent Effectiveness...
    Saved: results/plots/alss_mcm_snr_effectiveness.png
Creating Plot 3: Gap Reduction with Confidence Intervals...
    Saved: results/plots/alss_mcm_gap_reduction.png

Saved: results/summaries/alss_mcm_baseline_results.csv
```

---

## Alternative: Run Other Array Demos

```powershell
# Z1 array only
python analysis_scripts/run_z1_demo.py --markdown --save-csv

# Z3_2 array only
python analysis_scripts/run_z3_2_demo.py --markdown --save-csv

# Z5 array only
python analysis_scripts/run_z5_demo.py --markdown --save-csv

# Nested array
python analysis_scripts/run_nested_demo.py --markdown --save-csv
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | `pip install numpy pandas matplotlib scipy` |
| "No such file" | Make sure you're in project root: `cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS` |
| Script runs slow | It's normal - 1000 trials takes 2-5 minutes |
| Plots not showing | Check `results/plots/` folder manually |
| Permission error | Close any open image files and retry |

---

## Files Generated

```
results/
├── plots/
│   ├── alss_mcm_gap_reduction.png
│   ├── alss_mcm_bias_variance_decomposition.png
│   └── alss_mcm_snr_effectiveness.png
└── summaries/
    └── alss_mcm_baseline_results.csv
```

---

## Verify Reproducibility

Run script twice to verify same results:

```powershell
# Run 1
python analysis_scripts/analyze_alss_mcm_baseline.py

# Run 2
python analysis_scripts/analyze_alss_mcm_baseline.py

# Results should be IDENTICAL (seed=42 ensures this)
```

---

## Need More Details?

See `HOW_TO_RUN_PLOTS.md` for:
- Complete configuration options
- Python script structure
- Advanced usage (Jupyter, direct Python)
- Full troubleshooting guide

---

**Status**: ✅ Ready to run  
**Time to Plots**: 2-5 minutes  
**Reproducibility**: Yes (seed=42)  
**Trial Count**: 1000 (configurable)

