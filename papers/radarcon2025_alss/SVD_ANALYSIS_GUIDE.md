# SVD Analysis for MIMO Array Covariance Matrices

## Overview

This framework captures and analyzes singular value decompositions (SVDs) of physical and virtual covariance matrices to understand:
- **Model order determination** - Identify K signal sources from eigenvalue gaps
- **Condition number analysis** - Assess numerical stability (κ = σ_max / σ_min)
- **ALSS effectiveness** - Compare how ALSS improves Rv conditioning vs baseline

For Hermitian PSD matrices like Rx and Rv, **singular values = eigenvalues**.

---

## Quick Start

### 1. Run Benchmarks with SVD Capture

```powershell
# Capture SVD data during benchmarks (adds ~10% overhead)
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 --algs SpatialMUSIC,CoarrayMUSIC `
  --snr 0,5,10,15 --snapshots 64,128,256 --k 2 --delta 2 --trials 10 `
  --dump-svd `
  --out results\bench\svd_experiment.csv
```

**Generated files:**
- `results/svd/{array}_{alg}_N{N}_M{M}_snr{SNR}_trial{t}_Rx_svd.csv` (physical cov)
- `results/svd/{array}_{alg}_N{N}_M{M}_snr{SNR}_trial{t}_Rv_svd.csv` (virtual cov)

### 2. Analyze SVD Data

```powershell
# Generate plots and condition number tables
python tools\analyze_svd.py results\svd\
```

**Outputs** (in `results/svd/analysis/`):
- `condition_numbers.csv` - Full per-trial condition numbers
- `condition_numbers_summary.csv` - Grouped statistics (mean, std, min, max)
- `svd_Rx_*.png` - Physical covariance singular value spectra
- `svd_Rv_*.png` - Virtual covariance singular value spectra

---

## Interpreting Results

### Condition Number (κ)

**Lower is better** - indicates better signal/noise separation:

| κ Range | Interpretation |
|---------|---------------|
| κ < 100 | Excellent - Clean subspace separation |
| 100 < κ < 1000 | Good - Moderate conditioning |
| κ > 1000 | Poor - Numerical instability risk |

**Example from test run:**
```
                                 mean       std        min         max
array   M  SNR_dB cov_type
Z5(N=7) 64 10.0   Rv        40.2      4.8        34.8        43.8
                  Rx        99.8      1.6        98.2       101.5
```

✅ Virtual covariance (Rv) has **better conditioning** than physical (Rx) due to coarray processing!

### Singular Value Spectrum

**What to look for:**
1. **K large values** - Signal subspace (K sources)
2. **Sharp drop** - Clean separation (good for MUSIC)
3. **Noise floor** - Remaining small eigenvalues

**ALSS effect:** Reduces tail singular values → improves κ and signal/noise separation.

---

## Comparing ALSS ON vs OFF

### Step 1: Run baseline (ALSS OFF)
```powershell
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 --algs CoarrayMUSIC `
  --snr 10 --snapshots 64 --k 2 --delta 2 --trials 10 `
  --alss off --dump-svd `
  --out results\alss\baseline_svd.csv
```

### Step 2: Run with ALSS ON
```powershell
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 --algs CoarrayMUSIC `
  --snr 10 --snapshots 64 --k 2 --delta 2 --trials 10 `
  --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 --dump-svd `
  --out results\alss\alss_on_svd.csv
```

### Step 3: Compare
```powershell
# Copy ALSS ON files to separate folder
New-Item -ItemType Directory -Force results\svd_alss_on
Move-Item results\svd\*_Rv_svd.csv results\svd_alss_on\

# Analyze both
python tools\analyze_svd.py results\svd\         # Baseline
python tools\analyze_svd.py results\svd_alss_on\ # ALSS ON

# Compare condition numbers
Compare-Object (Import-Csv results\svd\analysis\condition_numbers_summary.csv) `
               (Import-Csv results\svd_alss_on\analysis\condition_numbers_summary.csv)
```

---

## Advanced: Programmatic Access

```python
import numpy as np
import pandas as pd
from pathlib import Path

# Load singular values
def load_svd(csv_path):
    return np.loadtxt(csv_path, delimiter=',').flatten()

# Compute condition number
def condition_number(sv):
    return sv[0] / max(sv[-1], 1e-12)

# Example: Load and analyze
sv_rx = load_svd("results/svd/Z5(N=7)_SpatialMUSIC_N7_M64_snr10.0_trial0_Rx_svd.csv")
sv_rv = load_svd("results/svd/Z5(N=7)_CoarrayMUSIC_N7_M64_snr10.0_trial0_Rv_svd.csv")

print(f"Rx κ = {condition_number(sv_rx):.2f}")
print(f"Rv κ = {condition_number(sv_rv):.2f}")
print(f"Rx spectrum: {sv_rx}")
print(f"Rv spectrum: {sv_rv}")
```

---

## Integration with Existing Analysis

### Add SVD metrics to existing CSV results

```python
import pandas as pd

# Load benchmark results
df = pd.read_csv("results/bench/svd_experiment.csv")

# For each trial, load corresponding SVD and add condition number
for idx, row in df.iterrows():
    svd_file = f"results/svd/{row['array']}_{row['alg']}_N{row['N']}_M{row['snapshots']}_snr{row['SNR_dB']}_trial{idx}_Rv_svd.csv"
    if Path(svd_file).exists():
        sv = np.loadtxt(svd_file, delimiter=',').flatten()
        df.loc[idx, 'Rv_cond'] = sv[0] / max(sv[-1], 1e-12)

# Save enhanced results
df.to_csv("results/bench/svd_experiment_enhanced.csv", index=False)
```

---

## Performance Notes

- **Overhead:** ~10% additional runtime (SVD is O(N³))
- **Storage:** ~200 bytes per trial per matrix
- **Large sweeps:** For 1000+ trials, consider sampling subset with `--trials 100`

---

## Troubleshooting

### No SVD files generated
```powershell
# Check if --dump-svd flag was used
python core\analysis_scripts\run_benchmarks.py --help | Select-String "dump-svd"

# Verify results/svd/ directory exists
Test-Path results\svd
```

### Parsing errors in analyze_svd.py
```powershell
# Check filename format (should match pattern)
Get-ChildItem results\svd\*.csv | Select-Object Name

# Ensure no special characters in array names
# If needed, edit tools/analyze_svd.py parse_filename() regex
```

### Condition number = inf
- **Cause:** Near-zero smallest singular value (rank deficiency)
- **Fix:** Increase diagonal loading in coarray_music.py (eps parameter)
- **Or:** Use more snapshots (M) to improve covariance estimation

---

## References

- **Paper:** Section 4.3 - Subspace stability and condition number analysis
- **Code:** 
  - `core/radarpy/algorithms/spatial_music.py` - Rx SVD
  - `core/radarpy/algorithms/coarray_music.py` - Rv SVD
  - `tools/analyze_svd.py` - Analysis and visualization
