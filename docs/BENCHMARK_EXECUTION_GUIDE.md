# Benchmark Execution Guide

**Date:** November 6, 2025  
**Author:** MIMO Geometry Analysis Team

## Overview

This guide explains how to run performance benchmarks for MIMO array geometries with proper virtual environment activation. Benchmarks evaluate DOA estimation accuracy across different array types with and without mutual coupling effects.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Virtual Environment Activation](#virtual-environment-activation)
3. [Running Benchmarks](#running-benchmarks)
4. [Mutual Coupling Benchmarks](#mutual-coupling-benchmarks)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.13.0** (managed via pyenv)
- **Virtual Environment:** `envs/mimo-geom-dev/`
- **Key Packages:**
  - NumPy >= 2.3.4
  - Pandas >= 2.3.3
  - Matplotlib >= 3.10.7
  - tabulate >= 0.9.0

### Verify Installation
```powershell
# Check Python version
python --version  # Should show: Python 3.13.0

# Check virtual environment exists
Test-Path .\envs\mimo-geom-dev\Scripts\activate.bat  # Should return: True
```

---

## Virtual Environment Activation

### Why Activation is Required

Benchmarks require virtual environment activation to ensure:
- ✅ **Consistent package versions** across all test runs
- ✅ **Isolated dependencies** from system Python
- ✅ **Reproducible results** for performance comparisons
- ✅ **Proper imports** for custom modules (radarpy.signal)

### Method 1: Batch File (Recommended)

**Always works** on Windows with no restrictions:

```batch
# Activate environment
.\activate_venv.bat

# Verify activation
where python
# Should show path containing: envs\mimo-geom-dev\Scripts\python.exe
```

### Method 2: PowerShell Script

**Requires** `RemoteSigned` execution policy:

```powershell
# Activate environment
.\activate_venv.ps1

# If blocked by execution policy, run once as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Method 3: Direct Activation

Manual activation without wrapper scripts:

```powershell
# PowerShell
.\envs\mimo-geom-dev\Scripts\Activate.ps1

# Command Prompt (CMD)
.\envs\mimo-geom-dev\Scripts\activate.bat
```

### Verify Activation Success

After activation, verify the environment:

```powershell
# Check Python location
python -c "import sys; print(sys.executable)"
# Expected: C:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS\envs\mimo-geom-dev\Scripts\python.exe

# Check package versions
python -c "import numpy as np; import pandas as pd; print(f'NumPy: {np.__version__}, Pandas: {pd.__version__}')"
# Expected: NumPy: 2.3.4, Pandas: 2.3.3
```

---

## Running Benchmarks

### Using Automated Benchmark Runners

The project provides **automated scripts** that handle venv activation:

#### PowerShell Runner

```powershell
# Basic benchmark (Z5 array, N=7, 100 trials)
.\run_benchmarks_with_venv.ps1

# Custom parameters
.\run_benchmarks_with_venv.ps1 -Arrays "ULA,Nested,Z5" -N 10 -Trials 200

# With mutual coupling enabled
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 100 -WithCoupling

# Pass additional arguments
.\run_benchmarks_with_venv.ps1 -AdditionalArgs "--snr 0,5,10,15,20 --save-csv"
```

#### Batch Runner

```batch
REM Basic benchmark (Z5 array, N=7, 100 trials)
run_benchmarks_with_venv.bat

REM Custom parameters: arrays, N, trials, [coupling]
run_benchmarks_with_venv.bat ULA 8 150

REM With mutual coupling
run_benchmarks_with_venv.bat Z5 7 100 coupling
```

### Manual Benchmark Execution

If you prefer manual control:

```powershell
# Step 1: Activate environment
.\activate_venv.bat

# Step 2: Run benchmark script
python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --trials 100

# Step 3: Deactivate when done
deactivate
```

---

## Mutual Coupling Benchmarks

### Overview

The MCM (Mutual Coupling Matrix) feature allows benchmarking with **realistic antenna coupling** effects:

- **Exponential Model:** Distance-based coupling decay
- **Toeplitz Model:** Symmetric coupling structure for ULAs
- **Measured Model:** Load coupling coefficients from CSV/NPY files

### Benchmark Scenarios

#### Scenario 1: Ideal vs. Coupled Performance

Compare DOA estimation with and without coupling:

```powershell
# Without coupling (baseline)
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 200

# With exponential coupling (c₁=0.3, α=0.5)
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 200 -WithCoupling
```

Expected results:
- **Ideal:** RMSE ≈ 0.14° (baseline)
- **Coupled:** RMSE ≈ 0.25° (+80% degradation)

#### Scenario 2: Coupling Strength Sweep

Test multiple coupling strengths:

```powershell
# Manual activation for batch processing
.\activate_venv.bat

# Loop through coupling strengths
foreach ($c1 in 0.1, 0.2, 0.3, 0.4, 0.5) {
    python core\analysis_scripts\run_benchmarks.py `
        --arrays Z5 `
        --N 7 `
        --trials 100 `
        --coupling exponential `
        --coupling-strength $c1 `
        --save-csv `
        --output-dir results/coupling_sweep_c$c1
}

deactivate
```

#### Scenario 3: Array Type Comparison with Coupling

Compare performance across array types under coupling:

```batch
REM ULA benchmark
run_benchmarks_with_venv.bat ULA 8 100 coupling

REM Nested array benchmark
run_benchmarks_with_venv.bat Nested 7 100 coupling

REM Z5 array benchmark
run_benchmarks_with_venv.bat Z5 7 100 coupling
```

### MCM Configuration Options

When using manual execution, specify coupling parameters:

```powershell
python core\analysis_scripts\run_benchmarks.py `
    --arrays Z5 `
    --N 7 `
    --trials 100 `
    --coupling exponential `           # Model: exponential, toeplitz, measured
    --coupling-strength 0.3 `          # c₁ coefficient (0.1-0.5)
    --coupling-decay 0.5 `             # α decay factor (0.3-1.0)
    --save-csv `
    --save-plots
```

---

## Advanced Usage

### Batch Processing Multiple Configurations

Create a PowerShell script for comprehensive benchmarks:

```powershell
# benchmark_sweep.ps1
.\activate_venv.bat

$arrays = @("ULA", "Nested", "Z5")
$sensors = @(7, 10, 15)
$snr_values = @(0, 5, 10, 15, 20)

foreach ($arr in $arrays) {
    foreach ($N in $sensors) {
        python core\analysis_scripts\run_benchmarks.py `
            --arrays $arr `
            --N $N `
            --trials 200 `
            --snr ($snr_values -join ',') `
            --save-csv `
            --output-dir "results/sweep_${arr}_N${N}"
    }
}

deactivate
Write-Host "Benchmark sweep complete!" -ForegroundColor Green
```

### Performance Profiling

Profile benchmark execution time:

```powershell
.\activate_venv.bat

Measure-Command {
    python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --trials 100
}

deactivate
```

### Parallel Execution

Run multiple benchmarks in parallel:

```powershell
# Start background jobs (PowerShell)
.\activate_venv.bat

Start-Job -ScriptBlock { python core\analysis_scripts\run_benchmarks.py --arrays ULA --N 8 --trials 100 --save-csv --output-dir results/job1 }
Start-Job -ScriptBlock { python core\analysis_scripts\run_benchmarks.py --arrays Nested --N 7 --trials 100 --save-csv --output-dir results/job2 }
Start-Job -ScriptBlock { python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --trials 100 --save-csv --output-dir results/job3 }

# Wait for completion
Get-Job | Wait-Job
Get-Job | Receive-Job
Get-Job | Remove-Job

deactivate
```

---

## Troubleshooting

### Issue 1: Virtual Environment Not Found

**Symptom:**
```
ERROR: Virtual environment not found at: envs\mimo-geom-dev\Scripts\activate.bat
```

**Solution:**
```powershell
# Verify venv directory exists
Test-Path .\envs\mimo-geom-dev\

# If missing, recreate venv
python -m venv envs\mimo-geom-dev

# Install dependencies
.\activate_venv.bat
pip install -r requirements.txt
```

### Issue 2: PowerShell Execution Policy

**Symptom:**
```
.\activate_venv.ps1 : File cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
# Option 1: Use batch file instead (no restrictions)
.\activate_venv.bat

# Option 2: Change execution policy (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Option 3: Bypass for single execution
PowerShell -ExecutionPolicy Bypass -File .\activate_venv.ps1
```

### Issue 3: Import Errors After Activation

**Symptom:**
```
ModuleNotFoundError: No module named 'radarpy'
```

**Solution:**
```powershell
# Verify packages installed
.\activate_venv.bat
pip list | Select-String "numpy|pandas|matplotlib"

# If missing, reinstall
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"
# Should include: C:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS
```

### Issue 4: Benchmark Script Not Found

**Symptom:**
```
ERROR: Could not find benchmark script at: core\analysis_scripts\run_benchmarks.py
```

**Solution:**
```powershell
# Check if benchmark script exists
Test-Path .\core\analysis_scripts\run_benchmarks.py

# If missing, use demo scripts instead
python analysis_scripts\run_z5_demo.py --N 7 --d 1.0 --markdown
```

### Issue 5: Inconsistent Results

**Symptom:** Benchmark results vary significantly between runs

**Solution:**
```powershell
# Ensure venv is activated (checks Python location)
.\activate_venv.bat
python -c "import sys; print(sys.executable)"

# Use more trials for stable results
.\run_benchmarks_with_venv.ps1 -Trials 500

# Set random seed in benchmark script (for reproducibility)
python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --trials 100 --seed 42
```

---

## Best Practices

### ✅ Do's

1. **Always activate venv** before running benchmarks
2. **Use automated runners** (`run_benchmarks_with_venv.*`) for consistency
3. **Save results** with `--save-csv` and `--save-plots`
4. **Document parameters** in output directory names
5. **Run multiple trials** (≥100) for statistical significance
6. **Deactivate venv** after completing batch jobs

### ❌ Don'ts

1. **Don't mix** system Python and venv Python in same session
2. **Don't modify** core modules without deactivating venv first
3. **Don't run** benchmarks without venv activation
4. **Don't use** too few trials (<20) for performance claims
5. **Don't ignore** package version warnings

---

## Quick Reference

### Activation Commands

| Method | Command | Reliability | Notes |
|--------|---------|-------------|-------|
| Batch wrapper | `.\activate_venv.bat` | ⭐⭐⭐⭐⭐ | **Recommended** - Always works |
| PowerShell wrapper | `.\activate_venv.ps1` | ⭐⭐⭐⭐ | May require execution policy change |
| Direct batch | `.\envs\mimo-geom-dev\Scripts\activate.bat` | ⭐⭐⭐⭐⭐ | Manual alternative |
| Direct PowerShell | `.\envs\mimo-geom-dev\Scripts\Activate.ps1` | ⭐⭐⭐ | May be blocked |

### Benchmark Runner Commands

```powershell
# Basic usage
.\run_benchmarks_with_venv.ps1
.\run_benchmarks_with_venv.bat

# With parameters
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 100
run_benchmarks_with_venv.bat Z5 7 100

# With coupling
.\run_benchmarks_with_venv.ps1 -WithCoupling
run_benchmarks_with_venv.bat Z5 7 100 coupling
```

### Verification Commands

```powershell
# Python version
python --version

# Python location
python -c "import sys; print(sys.executable)"

# Package versions
python -c "import numpy as np; import pandas as pd; print(f'NumPy: {np.__version__}, Pandas: {pd.__version__}')"
```

---

## Examples

### Example 1: Quick Test Run

```powershell
# One-line test
.\run_benchmarks_with_venv.ps1 -Arrays "ULA" -N 5 -Trials 50

# Expected output:
# ========================================
# MIMO Array Benchmark Runner
# ========================================
# 
# ✓ Virtual environment activated
# Python: 3.13.0
# Location: C:\...\envs\mimo-geom-dev\Scripts\python.exe
# 
# Running benchmark:
#   python core\analysis_scripts\run_benchmarks.py --arrays ULA --N 5 --trials 50
# ...
# Benchmark Complete!
```

### Example 2: Production Benchmark Suite

```powershell
# Create and run comprehensive benchmark suite
.\activate_venv.bat

# Z5 array performance sweep
$N_values = 7, 10, 15, 20
$trials = 200

foreach ($N in $N_values) {
    Write-Host "Running Z5 benchmark for N=$N..." -ForegroundColor Cyan
    
    # Without coupling
    python core\analysis_scripts\run_benchmarks.py `
        --arrays Z5 `
        --N $N `
        --trials $trials `
        --snr 0,5,10,15,20 `
        --save-csv `
        --output-dir "results/production/Z5_N${N}_ideal"
    
    # With exponential coupling
    python core\analysis_scripts\run_benchmarks.py `
        --arrays Z5 `
        --N $N `
        --trials $trials `
        --snr 0,5,10,15,20 `
        --coupling exponential `
        --coupling-strength 0.3 `
        --save-csv `
        --output-dir "results/production/Z5_N${N}_coupled"
}

deactivate
Write-Host "Production benchmark suite complete!" -ForegroundColor Green
```

### Example 3: Comparative Analysis

```batch
REM Compare all array types with same configuration
REM Run from Command Prompt (CMD)

run_benchmarks_with_venv.bat ULA 8 150 > results\ULA_results.txt
run_benchmarks_with_venv.bat Nested 7 150 > results\Nested_results.txt
run_benchmarks_with_venv.bat Z5 7 150 > results\Z5_results.txt

echo Comparative analysis complete!
echo Results saved to: results\*_results.txt
```

---

## Additional Resources

- **MCM Implementation Guide:** `docs/MUTUAL_COUPLING_GUIDE.md`
- **MCM Summary:** `docs/MCM_IMPLEMENTATION_SUMMARY.md`
- **Project README:** `README.md`
- **Requirements:** `requirements.txt`

---

**Last Updated:** November 6, 2025  
**Version:** 1.0.0  
**Contact:** MIMO Geometry Analysis Team
