# Metrics Quick Reference

## The 5 Metrics Dictionary

```python
metrics = {
    'RMSE_degrees': float,      # Root Mean Square Error (°)
    'RMSE_CRB_ratio': float,    # Efficiency vs. theoretical bound
    'Resolution_Rate': float,   # Success percentage (0-100%)
    'Bias_degrees': float,      # Systematic error (°, signed)
    'Runtime_ms': float         # Mean execution time (ms)
}
```

## Quick Usage

### One-Line Computation

```python
from core.radarpy.signal.metrics import compute_scenario_metrics

metrics = compute_scenario_metrics(
    estimated_trials,  # List of estimated DOA arrays
    true_doas,         # Ground truth DOAs
    positions,         # Sensor positions
    wavelength,        # Signal wavelength
    snr_db,           # SNR in dB
    snapshots,        # Number of snapshots
    coupling_matrix,  # Optional: None or (N × N) matrix
    resolution_threshold=3.0,  # Default: 3°
    runtimes_ms=runtimes      # Optional: list of runtimes
)
```

### CLI Usage

```powershell
# Baseline
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100

# With coupling
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100 --with-coupling --coupling-strength 0.3

# Save results
python core\analysis_scripts\run_scenario_with_metrics.py --array Z5 --snr 10 --trials 100 --save-csv --save-json
```

## Interpretation Guidelines

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **RMSE** | < 0.5° | 0.5° - 2° | 2° - 5° | > 5° |
| **RMSE/CRB** | 1.0 - 1.5 | 1.5 - 3.0 | 3.0 - 10 | > 10 |
| **Resolution** | > 95% | 80% - 95% | 50% - 80% | < 50% |
| **Bias** | < 0.1° | 0.1° - 0.5° | 0.5° - 2° | > 2° |
| **Runtime** | < 10 ms | 10 - 50 ms | 50 - 200 ms | > 200 ms |

## Programmatic Access

```python
# Access individual metrics
rmse = metrics['RMSE_degrees']
efficiency = metrics['RMSE_CRB_ratio']
success_rate = metrics['Resolution_Rate']
bias = metrics['Bias_degrees']
runtime = metrics['Runtime_ms']

# Pretty print
from core.radarpy.signal.metrics import print_metrics_summary
print_metrics_summary(metrics, "My Scenario")

# Export to CSV
import pandas as pd
pd.DataFrame([metrics]).to_csv('results.csv', index=False)

# Export to JSON
import json
with open('results.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Functions Summary

```python
# Individual metric functions
compute_rmse(estimated, true)                          # → RMSE in degrees
compute_bias(estimated, true)                          # → Bias in degrees
compute_crb(pos, λ, doas, snr, M, C)                  # → CRB std array
compute_resolution_rate(trials, true, threshold)       # → Percentage
run_trial_with_timing(func, *args, **kwargs)          # → (result, time_ms)

# All-in-one function
compute_scenario_metrics(trials, true, pos, λ, snr, M, C, threshold, times)  # → dict

# Display function
print_metrics_summary(metrics, name, show_crb)         # → Pretty output
```

## Example Output

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

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `RMSE_CRB_ratio < 1.0` | Statistical fluctuation | Normal (good performance) |
| `Resolution_Rate = 0%` | Detection failure | Increase SNR/snapshots |
| `RuntimeWarning: CRB` | Singular matrix | Check array geometry |
| Large `Runtime_ms` | Dense scan grid | Reduce grid resolution |

## Statistical Recommendations

- **Minimum trials:** 100 (basic)
- **Recommended:** 500 (stable metrics)
- **Publication:** 1000+ (high confidence)
- **Resolution threshold:** 3° (standard), 1° (conservative), 5° (relaxed)

## References

- **Full Guide:** `docs/METRICS_GUIDE.md`
- **Implementation:** `core/radarpy/signal/metrics.py`
- **Example Runner:** `core/analysis_scripts/run_scenario_with_metrics.py`

---

**Status:** ✅ Ready for use  
**Date:** November 6, 2025
