# ALSS Extended Sweep Guide

**Created:** November 4, 2025  
**Purpose:** Comprehensive ALSS SNR sweeps with multiple delta values and custom DOAs

---

## 🚀 Quick Start

### **1. Run Basic SNR Sweep**
```powershell
# Default: SNR=[-5,0,5,10,15]dB, Delta=[2,13]°, M=64, 100 trials/point
.\scripts\sweep_alss_snr.ps1
```

### **2. Custom Parameters**
```powershell
# Custom SNR and delta values
.\scripts\sweep_alss_snr.ps1 -SNRs @(-10, 0, 10, 20) -Deltas @(1, 5, 10, 15) -Trials 200

# Different snapshot counts
.\scripts\sweep_alss_snr.ps1 -Snapshots 128 -Trials 50

# Custom output directory
.\scripts\sweep_alss_snr.ps1 -OutputDir "results\custom_sweep"
```

---

## 📊 Usage Examples

### **Example 1: Standard Sweep (Default)**
```powershell
.\scripts\sweep_alss_snr.ps1
```
**Output:**
- `results/alss/baseline_Z5_M64_d2_snr*.csv` (5 files)
- `results/alss/alss_Z5_M64_d2_snr*.csv` (5 files)
- `results/alss/baseline_Z5_M64_d13_snr*.csv` (5 files)
- `results/alss/alss_Z5_M64_d13_snr*.csv` (5 files)
- `results/alss/all_runs.csv` (merged file)

**Total runs:** 20 (5 SNRs × 2 deltas × 2 modes)

### **Example 2: Fine-Grained SNR Sweep**
```powershell
.\scripts\sweep_alss_snr.ps1 -SNRs @(-5, -2, 0, 2, 5, 7, 10, 12, 15) -Trials 50
```
**Total runs:** 36 (9 SNRs × 2 deltas × 2 modes)

### **Example 3: Many Delta Values**
```powershell
.\scripts\sweep_alss_snr.ps1 -Deltas @(1, 2, 5, 10, 13, 20) -Trials 50
```
**Total runs:** 60 (5 SNRs × 6 deltas × 2 modes)

---

## 🎯 Custom DOA Angles

### **Use Case: Non-Symmetric Sources**

Instead of symmetric ±Δ/2, specify exact angles:

```powershell
# Example: Sources at -7° and +12° (non-symmetric)
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 --algs CoarrayMUSIC `
  --lambda_factor 2.0 --snr 5 --snapshots 64 --k 2 `
  --doas "-7;12" --trials 100 `
  --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
  --out results\alss\Z5_custom_DOAs_m7p12.csv
```

### **Multiple Custom Scenarios**
```powershell
# Test different asymmetric pairs
$doa_pairs = @("-10;5", "-15;8", "-20;15")

foreach ($doas in $doa_pairs) {
    $name = $doas.Replace(";", "_").Replace("-", "m")
    
    # Baseline
    python core\analysis_scripts\run_benchmarks.py `
      --arrays Z5 --N 7 --algs CoarrayMUSIC `
      --lambda_factor 2.0 --snr 5 --snapshots 64 --k 2 `
      --doas "$doas" --trials 100 --alss off `
      --out results\alss\baseline_custom_$name.csv
    
    # ALSS
    python core\analysis_scripts\run_benchmarks.py `
      --arrays Z5 --N 7 --algs CoarrayMUSIC `
      --lambda_factor 2.0 --snr 5 --snapshots 64 --k 2 `
      --doas "$doas" --trials 100 `
      --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
      --out results\alss\alss_custom_$name.csv
}
```

---

## 📈 Plotting Results

### **1. Plot Individual Delta Values**
```powershell
python tools\plot_alss_sweep.py results\alss\all_runs.csv
```
**Generates:**
- `results/alss/figures/alss_rmse_delta2.pdf`
- `results/alss/figures/alss_rmse_delta13.pdf`
- `results/alss/figures/alss_resolve_delta2.pdf`
- `results/alss/figures/alss_resolve_delta13.pdf`
- `results/alss/figures/alss_combined_all_deltas.pdf` (side-by-side comparison)

### **2. Analyze Results**
```powershell
# Summary statistics
python tools\analyze_alss_results.py `
  results\alss\baseline_Z5_M64_d13_snr5.csv `
  results\alss\alss_Z5_M64_d13_snr5.csv

# Check specific trials
python tools\check_baseline_samples.py results\alss\baseline_Z5_M64_d13_snr5.csv
```

---

## 🔧 Advanced Sweep Options

### **Sweep Snapshots (M)**
```powershell
$snapshots = @(32, 64, 128, 256)
$snrs = @(0, 5, 10)

foreach ($M in $snapshots) {
  foreach ($snr in $snrs) {
    # Baseline
    python core\analysis_scripts\run_benchmarks.py `
      --arrays Z5 --N 7 --algs CoarrayMUSIC `
      --lambda_factor 2.0 --snr $snr --snapshots $M `
      --k 2 --delta 13 --trials 100 `
      --alss off `
      --out results\alss\baseline_Z5_M${M}_d13_snr${snr}.csv
    
    # ALSS
    python core\analysis_scripts\run_benchmarks.py `
      --arrays Z5 --N 7 --algs CoarrayMUSIC `
      --lambda_factor 2.0 --snr $snr --snapshots $M `
      --k 2 --delta 13 --trials 100 `
      --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
      --out results\alss\alss_Z5_M${M}_d13_snr${snr}.csv
  }
}
```

### **Compare Multiple Array Types**
```powershell
$arrays = @("ULA", "Z4", "Z5", "Z6")
$snr = 5
$delta = 13

foreach ($arr in $arrays) {
  # Baseline
  python core\analysis_scripts\run_benchmarks.py `
    --arrays $arr --N 7 --algs CoarrayMUSIC `
    --lambda_factor 2.0 --snr $snr --snapshots 64 `
    --k 2 --delta $delta --trials 100 `
    --alss off `
    --out results\alss\baseline_${arr}_snr${snr}.csv
  
  # ALSS
  python core\analysis_scripts\run_benchmarks.py `
    --arrays $arr --N 7 --algs CoarrayMUSIC `
    --lambda_factor 2.0 --snr $snr --snapshots 64 `
    --k 2 --delta $delta --trials 100 `
    --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
    --out results\alss\alss_${arr}_snr${snr}.csv
}
```

---

## 📋 Parameter Reference

### **Sweep Script Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SNRs` | int[] | `@(-5,0,5,10,15)` | SNR values in dB |
| `Deltas` | int[] | `@(2,13)` | Angular separation in degrees |
| `Trials` | int | `100` | Trials per condition |
| `Snapshots` | int | `64` | Number of snapshots (M) |
| `OutputDir` | string | `results\alss` | Output directory |

### **Benchmark CLI Flags**
| Flag | Default | Description |
|------|---------|-------------|
| `--arrays` | - | Array type (ULA, Z4, Z5, Z6) |
| `--N` | - | Number of sensors |
| `--algs` | `SpatialMUSIC` | Algorithms (comma-separated) |
| `--snr` | `0` | SNR in dB |
| `--snapshots` | `64` | Number of snapshots |
| `--k` | `2` | Number of sources |
| `--delta` | `2` | Angular separation (degrees) |
| `--doas` | `None` | Custom DOAs (semicolon-separated, e.g. "-7;12") |
| `--trials` | `100` | Number of Monte Carlo trials |
| `--alss` | `off` | ALSS enable (on/off) |
| `--alss-mode` | `zero` | ALSS mode (zero/ar1) |
| `--alss-tau` | `1.0` | ALSS shrinkage strength |
| `--alss-coreL` | `3` | Protected lags (0..coreL) |

---

## 💡 Tips & Best Practices

### **Performance**
- Use `--trials 50` for quick tests, `--trials 200+` for publication
- Parallel runs: Open multiple PowerShell windows for different deltas
- Expected runtime: ~2-3 seconds per trial (depends on M and algorithm)

### **Data Organization**
- Keep sweep results in separate directories: `results/alss/sweep1/`, `results/alss/sweep2/`
- Name files descriptively: include array, M, delta, SNR in filename
- Always merge into `all_runs.csv` for plotting

### **Plotting**
- Generate plots after each sweep to catch issues early
- Use PDF for LaTeX papers, PNG for presentations
- Combine multiple deltas in side-by-side plots for comparisons

### **Analysis**
- Check baseline performance first (should match theory)
- Look for trends: ALSS helps most at moderate SNR (5-10dB)
- Verify no degradation at high SNR/high M (sanity check)

---

## 🐛 Troubleshooting

### **Script Fails with "Module not found"**
```powershell
# Activate virtual environment first
.\envs\mimo-geom-dev\Scripts\Activate.ps1
python core\analysis_scripts\run_benchmarks.py ...
```

### **Plots Look Bad**
```powershell
# Ensure matplotlib backend is set correctly
# Check tools/plot_alss_sweep.py line 14: matplotlib.use('Agg')
```

### **CSV Merge Fails**
```powershell
# Check column names match across files
# All CSVs should have same header format
```

---

## 📚 Related Files

- **Sweep Script:** `scripts/sweep_alss_snr.ps1`
- **Benchmark Runner:** `core/analysis_scripts/run_benchmarks.py`
- **Plotting (Standard):** `tools/plot_alss_figures.py`
- **Plotting (Sweep):** `tools/plot_alss_sweep.py`
- **Analysis:** `tools/analyze_alss_results.py`
- **Paper Integration:** `papers/radarcon2025_alss/ALSS_INTEGRATION_GUIDE.md`
