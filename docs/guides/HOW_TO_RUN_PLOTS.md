# How to Run Python Code to Produce Plots

## Quick Start

### Option 1: Run the Main ALSS_MCM Analysis Script (Recommended)

**Command:**
```powershell
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
python analysis_scripts/analyze_alss_mcm_baseline.py
```

**What it does:**
- Runs 1000 Monte Carlo trials with seed=42 (reproducible)
- Tests 3 arrays: Z1, Z3_2, Z5
- Generates 3 publication-quality plots
- Saves results CSV

**Output:**
```
✅ Plot 1: results/plots/alss_mcm_gap_reduction.png
✅ Plot 2: results/plots/alss_mcm_bias_variance_decomposition.png
✅ Plot 3: results/plots/alss_mcm_snr_effectiveness.png
✅ CSV:    results/summaries/alss_mcm_baseline_results.csv
```

**Time:** ~2-5 minutes (depending on system)

---

### Option 2: Run Other Available Demo Scripts

#### Run Z1 Array Demo:
```powershell
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
python analysis_scripts/run_z1_demo.py --markdown --save-csv
```

#### Run Z3_2 Array Demo:
```powershell
python analysis_scripts/run_z3_2_demo.py --markdown --save-csv
```

#### Run Z5 Array Demo:
```powershell
python analysis_scripts/run_z5_demo.py --markdown --save-csv
```

#### Run Nested Array Demo:
```powershell
python analysis_scripts/run_nested_demo.py --markdown --save-csv
```

---

## Step-by-Step Guide

### Step 1: Open PowerShell Terminal
- Press `Ctrl + Alt + T` or open PowerShell manually
- Or use VS Code Terminal (`Ctrl + backtick`)

### Step 2: Navigate to Project Directory
```powershell
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
```

### Step 3: Activate Virtual Environment (Optional but Recommended)
```powershell
# Windows PowerShell
.\mimo-geom-dev\Scripts\Activate.ps1

# Or batch file
.\mimo-geom-dev\Scripts\activate.bat
```

### Step 4: Run the Analysis Script
```powershell
python analysis_scripts/analyze_alss_mcm_baseline.py
```

### Step 5: Check Generated Plots
Plots are saved to: `results/plots/`

You can view them with:
```powershell
# Open the plots folder
explorer results\plots
```

---

## Configuration Options

### Modify Trial Count (in Python)

**Edit file:** `analysis_scripts/analyze_alss_mcm_baseline.py`

Find this line (around line 48):
```python
NUM_TRIALS = 1000
```

Change to desired value:
- `NUM_TRIALS = 100` → Faster run (~1 min)
- `NUM_TRIALS = 500` → Medium run (~3 min)
- `NUM_TRIALS = 1000` → Full run (~5 min)
- `NUM_TRIALS = 5000` → Extended run (~20 min)

### Modify Random Seed (for reproducibility)

Find this line (around line 52):
```python
REPRODUCIBILITY_SEED = 42
```

Change the seed value (any integer):
- `REPRODUCIBILITY_SEED = 42` → Current (produces known results)
- `REPRODUCIBILITY_SEED = 123` → Different seed (different random numbers, but still reproducible)

---

## Complete Example Workflow

### Full Reproducibility Test (Run Twice to Verify)

**Run 1:**
```powershell
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
python analysis_scripts/analyze_alss_mcm_baseline.py > run1.txt
```

**Run 2:**
```powershell
python analysis_scripts/analyze_alss_mcm_baseline.py > run2.txt
```

**Compare Results:**
```powershell
# Results should be identical (proving reproducibility with seed=42)
Compare-Object (Get-Content run1.txt) (Get-Content run2.txt)
```

---

## Understanding the Script Output

When you run the script, you'll see:

```
###########################################################################
#####                ALSS+MCM BASELINE EXPERIMENTAL VALIDATION           #####
###########################################################################

Testing Array: Z1
  Running 1000 trials...
    Completed 1000/1000 trials
  
  Results:
    Condition 1 (No MCM, No ALSS):  7.215° ± 18.836°
    Condition 2 (No MCM, ALSS ON):  6.493° ± 16.952° [SIMULATED]
    Condition 3 (MCM ON, No ALSS):  7.273° ± 19.119°
    Condition 4 (MCM ON, ALSS ON):  7.255° ± 17.207° [SIMULATED]
    Gap Reduction: 30.0%

[Similar output for Z3_2 and Z5]

===========================================================================
GENERATING PUBLICATION PLOTS
===========================================================================

Creating Plot 1: Bias-Variance Decomposition...
    Saved: results/plots/alss_mcm_bias_variance_decomposition.png
Creating Plot 2: SNR-Dependent Effectiveness...
    Saved: results/plots/alss_mcm_snr_effectiveness.png
Creating Plot 3: Gap Reduction with Confidence Intervals...
    Saved: results/plots/alss_mcm_gap_reduction.png

Saved: results/summaries/alss_mcm_baseline_results.csv
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Install dependencies
```powershell
pip install numpy pandas matplotlib scipy
```

### Issue: "FileNotFoundError: geometry_processors"
**Solution:** Make sure you're running from the project root directory
```powershell
# CORRECT:
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
python analysis_scripts/analyze_alss_mcm_baseline.py

# WRONG (don't do this):
cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS\analysis_scripts
python analyze_alss_mcm_baseline.py
```

### Issue: Script runs but no plots appear
**Solution:** Check the output directory
```powershell
ls results/plots/
```

If files exist but aren't displaying, open directly:
```powershell
explorer results\plots\alss_mcm_gap_reduction.png
```

### Issue: "Permission denied" when saving plots
**Solution:** Check folder permissions or close existing image files
```powershell
# Remove old plots and try again
rm results\plots\*.png
python analysis_scripts/analyze_alss_mcm_baseline.py
```

---

## Python Script Structure (For Reference)

The script performs these steps:

```python
# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Set reproducibility
REPRODUCIBILITY_SEED = 42
np.random.seed(REPRODUCIBILITY_SEED)
NUM_TRIALS = 1000

# Step 3: Initialize arrays
from geometry_processors.z1_processor import Z1ArrayProcessor
processor_z1 = Z1ArrayProcessor(N=7, d=1.0)

# Step 4: Run experiments (1000 trials)
for trial in range(NUM_TRIALS):
    # Generate signal, add noise, estimate DOA
    # Store RMSE results

# Step 5: Generate plots
import matplotlib.pyplot as plt
plt.figure()
# ... plot code ...
plt.savefig('results/plots/alss_mcm_gap_reduction.png')

# Step 6: Save CSV
df.to_csv('results/summaries/alss_mcm_baseline_results.csv')
```

---

## Running Plots Programmatically in Python

### Option 1: Direct Python Command
```powershell
python -c "exec(open('analysis_scripts/analyze_alss_mcm_baseline.py').read())"
```

### Option 2: Python Interactive Mode
```powershell
python
>>> import sys
>>> sys.path.append('.')
>>> exec(open('analysis_scripts/analyze_alss_mcm_baseline.py').read())
>>> exit()
```

### Option 3: In Jupyter Notebook
```python
%cd c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
%run analysis_scripts/analyze_alss_mcm_baseline.py
```

---

## View Generated Plots

### Method 1: Open in Default Image Viewer
```powershell
# Single plot
start results\plots\alss_mcm_gap_reduction.png

# All plots
explorer results\plots
```

### Method 2: View in Python
```powershell
python
>>> from PIL import Image
>>> import matplotlib.pyplot as plt
>>> img = Image.open('results/plots/alss_mcm_gap_reduction.png')
>>> plt.imshow(img)
>>> plt.show()
```

### Method 3: Open in VS Code
In VS Code, right-click on the PNG file and select "Open Preview"

---

## Summary

| Task | Command |
|------|---------|
| **Generate plots** | `python analysis_scripts/analyze_alss_mcm_baseline.py` |
| **View plots** | `explorer results\plots` |
| **Check results CSV** | `cat results\summaries\alss_mcm_baseline_results.csv` |
| **Activate venv** | `.\mimo-geom-dev\Scripts\Activate.ps1` |
| **Install dependencies** | `pip install numpy pandas matplotlib scipy` |
| **Run specific array** | `python analysis_scripts/run_z5_demo.py` |

---

## Key Files

| File | Purpose |
|------|---------|
| `analyze_alss_mcm_baseline.py` | Main analysis script (generates 3 plots + 1000 trials) |
| `run_z1_demo.py` | Z1 array demo |
| `run_z3_2_demo.py` | Z3_2 array demo |
| `run_z5_demo.py` | Z5 array demo |
| `results/plots/` | Output directory for plots |
| `results/summaries/` | Output directory for CSV results |

---

**Created**: November 9, 2025  
**Version**: 1.0  
**Status**: Ready to use

