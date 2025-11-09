# Comprehensive Metrics Guide

## Overview

This guide explains how to compute and use the **5 comprehensive metrics** for evaluating DOA estimation performance across experimental scenarios.

**Date:** November 6, 2025  
**Author:** MIMO Geometry Analysis Team

---

## The 5 Core Metrics

### 1. **RMSE_degrees** (Root Mean Square Error)
- **Definition:** Average estimation error magnitude
- **Units:** Degrees
- **Range:** 0° to ∞ (lower is better)
- **Formula:** $\text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=1}^T \frac{1}{K}\sum_{k=1}^K (\hat{\theta}_{k,t} - \theta_k)^2}$
- **Interpretation:**
  - < 0.5°: Excellent performance
  - 0.5° - 2°: Good performance
  - 2° - 5°: Acceptable performance
  - > 5°: Poor performance

### 2. **RMSE_CRB_ratio** (Efficiency Ratio)
- **Definition:** How close RMSE is to theoretical lower bound (Cramér-Rao Bound)
- **Units:** Dimensionless ratio
- **Range:** 1.0 to ∞ (closer to 1 is better)
- **Formula:** $\text{Efficiency} = \frac{\text{RMSE}}{\sqrt{\text{CRB}}}$
- **Interpretation:**
  - 1.0 - 1.5: Near-optimal performance
  - 1.5 - 3.0: Good efficiency
  - 3.0 - 10: Acceptable efficiency
  - > 10: Inefficient estimator
- **Note:** Can be < 1.0 if RMSE computed over finite trials (stochastic fluctuation)

### 3. **Resolution_Rate** (Percentage)
- **Definition:** Percentage of trials where all sources are correctly resolved
- **Units:** Percentage (0-100%)
- **Range:** 0% to 100% (higher is better)
- **Threshold:** Default 3° (configurable)
- **Formula:** $\text{Resolution Rate} = \frac{\text{Resolved Trials}}{\text{Total Trials}} \times 100\%$
- **Interpretation:**
  - > 95%: Excellent resolution
  - 80% - 95%: Good resolution
  - 50% - 80%: Marginal resolution
  - < 50%: Poor resolution

### 4. **Bias_degrees** (Systematic Error)
- **Definition:** Mean signed error (systematic bias)
- **Units:** Degrees
- **Range:** -∞ to +∞ (closer to 0 is better)
- **Formula:** $\text{Bias} = \frac{1}{T}\sum_{t=1}^T \frac{1}{K}\sum_{k=1}^K (\hat{\theta}_{k,t} - \theta_k)$
- **Interpretation:**
  - < 0.1°: Negligible bias
  - 0.1° - 0.5°: Small bias
  - 0.5° - 2°: Moderate bias
  - > 2°: Significant bias
- **Note:** Positive bias = overestimation, Negative bias = underestimation

### 5. **Runtime_ms** (Computational Efficiency)
- **Definition:** Mean execution time per trial
- **Units:** Milliseconds
- **Range:** 0 ms to ∞ (lower is better)
- **Measurement:** Uses `time.perf_counter()` for high precision
- **Interpretation:**
  - < 10 ms: Very fast (real-time capable)
  - 10 - 50 ms: Fast
  - 50 - 200 ms: Moderate speed
  - > 200 ms: Slow

---

## Usage Examples

### Example 1: Basic Metrics Computation

```python
import numpy as np
from core.radarpy.signal.metrics import compute_scenario_metrics

# Run Monte Carlo trials
estimated_trials = []
runtimes = []

for trial in range(100):
    start = time.perf_counter()
    
    # Your DOA estimation code
    est_doas = run_music(Rxx, positions, wavelength, scan_grid, K)
    
    runtime_ms = (time.perf_counter() - start) * 1000
    estimated_trials.append(est_doas)
    runtimes.append(runtime_ms)

# Compute all metrics
metrics = compute_scenario_metrics(
    estimated_doas_trials=estimated_trials,
    true_doas=np.array([15.0, -20.0]),
    sensor_positions=positions,
    wavelength=1.0,
    snr_db=10.0,
    snapshots=256,
    coupling_matrix=None,  # Or your coupling matrix
    resolution_threshold=3.0,
    runtimes_ms=runtimes
)

# Access metrics
print(f"RMSE: {metrics['RMSE_degrees']:.4f}°")
print(f"RMSE/CRB: {metrics['RMSE_CRB_ratio']:.2f}x")
print(f"Resolution: {metrics['Resolution_Rate']:.1f}%")
print(f"Bias: {metrics['Bias_degrees']:.4f}°")
print(f"Runtime: {metrics['Runtime_ms']:.2f} ms")
```

### Example 2: Using the Scenario Runner Script

```powershell
# Activate virtual environment
.\mimo-geom-dev\Scripts\Activate.ps1

# Run baseline scenario (no coupling)
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100

# Run with mutual coupling
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100 --with-coupling --coupling-strength 0.3

# Custom DOAs
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100 --doas 10 -15 25

# Save results
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100 --save-csv --save-json
```

### Example 3: Using Timing Wrapper

```python
from core.radarpy.signal.metrics import run_trial_with_timing

# Single trial with automatic timing
est_doas, runtime_ms = run_trial_with_timing(
    run_music,
    Rxx=Rxx,
    sensor_positions=positions,
    wavelength=wavelength,
    scan_grid=scan_grid,
    K=K,
    coupling_matrix=C
)

print(f"Estimated: {est_doas}")
print(f"Runtime: {runtime_ms:.2f} ms")
```

### Example 4: Pretty Printing Results

```python
from core.radarpy.signal.metrics import print_metrics_summary

# After computing metrics
print_metrics_summary(
    metrics, 
    scenario_name="Z5 Array @ 10dB SNR with Coupling",
    show_crb_comparison=True
)
```

Output:
```
============================================================
  Z5 Array @ 10dB SNR with Coupling
============================================================
  RMSE:                 1.2345°
  RMSE/CRB Ratio:         2.34x
  Efficiency:            42.7%
  Resolution Rate:       85.0%
  Bias:                 0.1234°
  Runtime:               12.34 ms
============================================================
```

---

## Integration with Experimental Scenarios

### Scenario 1: Baseline Performance

```python
# No coupling - establish baseline
metrics_baseline = run_scenario(
    array_type='Z5',
    snr_db=10.0,
    num_trials=500,
    with_coupling=False
)
```

**Expected Metrics:**
- RMSE: 0.3° - 1.0° (depends on SNR)
- RMSE/CRB: 1.2 - 2.0 (near-optimal)
- Resolution: > 95%
- Bias: < 0.1°
- Runtime: 5-15 ms

### Scenario 2: Mutual Coupling Impact

```python
# With coupling - measure degradation
metrics_coupled = run_scenario(
    array_type='Z5',
    snr_db=10.0,
    num_trials=500,
    with_coupling=True,
    coupling_strength=0.3
)

# Compare degradation
rmse_degradation = (metrics_coupled['RMSE_degrees'] - metrics_baseline['RMSE_degrees']) / metrics_baseline['RMSE_degrees'] * 100
print(f"RMSE degradation: {rmse_degradation:.1f}%")
```

**Expected Impact:**
- RMSE increase: 50% - 300%
- RMSE/CRB increase: 2x - 5x
- Resolution drop: 10% - 40%
- Bias increase: Moderate
- Runtime: Similar (±10%)

### Scenario 3: SNR Sweep Analysis

```python
snr_range = np.arange(-5, 21, 5)  # -5 to 20 dB
results = []

for snr_db in snr_range:
    metrics = run_scenario(
        array_type='Z5',
        snr_db=snr_db,
        num_trials=500,
        with_coupling=True,
        coupling_strength=0.3
    )
    results.append({
        'SNR_dB': snr_db,
        **metrics
    })

# Create DataFrame for analysis
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

---

## Output Format

### CSV Export

```csv
RMSE_degrees,RMSE_CRB_ratio,Resolution_Rate,Bias_degrees,Runtime_ms
1.2345,2.3456,85.0,0.1234,12.34
```

### JSON Export

```json
{
  "RMSE_degrees": 1.2345,
  "RMSE_CRB_ratio": 2.3456,
  "Resolution_Rate": 85.0,
  "Bias_degrees": 0.1234,
  "Runtime_ms": 12.34
}
```

### Markdown Table

```markdown
| Array | SNR (dB) | RMSE (°) | RMSE/CRB | Resolution (%) | Bias (°) | Runtime (ms) |
|-------|----------|----------|----------|----------------|----------|--------------|
| Z5    | 10       | 1.2345   | 2.35     | 85.0           | 0.1234   | 12.34        |
| ULA   | 10       | 2.3456   | 3.45     | 75.0           | 0.2345   | 8.45         |
```

---

## Statistical Considerations

### Number of Trials

- **Minimum:** 100 trials for basic metrics
- **Recommended:** 500 trials for stable metrics
- **Publication:** 1000+ trials for high confidence

**Rule of thumb:** $\text{Standard Error} \approx \frac{\text{RMSE}}{\sqrt{N_{\text{trials}}}}$

### Resolution Threshold Selection

- **Conservative:** 1.0° (strict)
- **Standard:** 3.0° (typical radar applications)
- **Relaxed:** 5.0° (wide-beam systems)

**Recommendation:** Use 3° for general comparisons, report sensitivity analysis with multiple thresholds

### CRB Computation

The CRB implementation includes:
- **Signal model:** Narrowband far-field sources
- **Noise model:** Spatially white Gaussian
- **Coupling effects:** Incorporated into steering matrix
- **Numerical stability:** Validated for N ≤ 20, K ≤ 10

---

## Troubleshooting

### Issue: `RuntimeWarning: divide by zero in CRB`
**Cause:** Singular steering matrix (sources too close or collinear array)  
**Solution:** Increase source separation or check array geometry

### Issue: `Resolution_Rate = 0%`
**Cause:** Algorithm failing to detect correct number of sources  
**Solution:** Increase SNR, snapshots, or check threshold setting

### Issue: `RMSE_CRB_ratio < 1.0`
**Cause:** Finite-sample statistical fluctuation (valid)  
**Interpretation:** Your estimator is performing very well (within CRB)

### Issue: Very large `Runtime_ms` values
**Cause:** Inefficient grid search or large scan grid  
**Solution:** Reduce scan grid resolution or use coarser stepping

---

## API Reference

### `compute_scenario_metrics()`

**Signature:**
```python
def compute_scenario_metrics(
    estimated_doas_trials: List[np.ndarray],
    true_doas: np.ndarray,
    sensor_positions: np.ndarray,
    wavelength: float,
    snr_db: float,
    snapshots: int,
    coupling_matrix: Optional[np.ndarray] = None,
    resolution_threshold: float = 3.0,
    runtimes_ms: Optional[List[float]] = None
) -> Dict[str, float]:
```

**Returns:**
```python
{
    'RMSE_degrees': float,      # RMSE in degrees
    'RMSE_CRB_ratio': float,    # Efficiency metric
    'Resolution_Rate': float,   # Percentage (0-100)
    'Bias_degrees': float,      # Signed bias in degrees
    'Runtime_ms': float         # Mean runtime in milliseconds
}
```

### Individual Metric Functions

```python
# RMSE computation
rmse = compute_rmse(estimated_doas, true_doas)

# Bias computation
bias = compute_bias(estimated_doas, true_doas)

# CRB computation
crb_std = compute_crb(positions, wavelength, true_doas, snr_lin, snapshots, coupling_matrix)

# Resolution rate
rate = compute_resolution_rate(estimated_trials, true_doas, threshold_deg=3.0)

# Timing wrapper
est_doas, runtime_ms = run_trial_with_timing(run_music, Rxx, positions, wavelength, scan_grid, K)
```

---

## Example Workflow

```python
# 1. Setup
from core.radarpy.signal.doa_sim_core import simulate_snapshots, run_music
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.signal.metrics import compute_scenario_metrics, print_metrics_summary

# 2. Configuration
positions = np.array([0, 5, 8, 11, 14, 17, 21]) * 0.5  # Z5
true_doas = np.array([15.0, -20.0])
snr_db = 10.0
snapshots = 256
num_trials = 500

# 3. Generate coupling matrix
C = generate_mcm(len(positions), coupling_strength=0.3, model='exponential')

# 4. Run trials
estimated_trials = []
runtimes = []

for trial in range(num_trials):
    # Generate data
    X = simulate_snapshots(positions, 1.0, true_doas, snapshots, snr_db, C)
    Rxx = (X @ X.conj().T) / snapshots
    
    # Estimate with timing
    scan_grid = np.linspace(-90, 90, 1801)
    est, runtime = run_trial_with_timing(
        run_music, Rxx, positions, 1.0, scan_grid, 2, coupling_matrix=C
    )
    
    estimated_trials.append(est)
    runtimes.append(runtime)

# 5. Compute metrics
metrics = compute_scenario_metrics(
    estimated_trials, true_doas, positions, 1.0, snr_db, snapshots, C, 3.0, runtimes
)

# 6. Display results
print_metrics_summary(metrics, "Z5 Array with Coupling @ 10dB")

# 7. Save results
import pandas as pd
df = pd.DataFrame([metrics])
df.to_csv('results/summaries/scenario_metrics.csv', index=False)
```

---

## References

- **CRB Theory:** Stoica, P., & Nehorai, A. (1989). "MUSIC, maximum likelihood, and Cramér-Rao bound," IEEE Trans. ASSP
- **DOA Estimation:** Van Trees, H. L. (2002). "Optimum Array Processing," Wiley
- **Mutual Coupling:** Friedlander, B., & Weiss, A. J. (1991). "Direction finding in the presence of mutual coupling," IEEE Trans. AP

---

**Status:** ✅ Metrics module ready for comprehensive experimental evaluation!  
**Next Steps:** Run experimental scenarios with `run_scenario_with_metrics.py`
