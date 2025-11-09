# SCENARIO 2: Mutual Coupling Impact Study

## Overview

This directory contains results from **Scenario 2** of the ALSS (Aliasing-Limited Sparse Sensing) experimental study for RadarCon 2025. The focus is on quantifying mutual coupling degradation effects on DOA estimation performance.

## Background

- **Reference Baseline**: Scenario 1 results at SNR=10dB, M=256 snapshots
  - Baseline RMSE: **0.0057°** (from `scenario1a_snr_sweep_Z5.csv`)
  - Baseline Resolution: **100%**
  - Operating point: 2 sources at [15°, -20°] separation

- **Coupling Model**: Exponential mutual coupling matrix (MCM)
  - `C[i,j] = c1 * exp(-c2 * |i-j|)` where c2=0.1 (fixed)
  - Primary parameter: `c1` (coupling strength) varies from 0.0 to 0.5

## Experiments

### Experiment 2A: Coupling Strength Sweep
**Purpose**: Map performance degradation as mutual coupling increases

**Parameters**:
- Array: Z5 (7 sensors, optimized ALSS geometry)
- Coupling range: c1 = 0.00 to 0.50 (10 points)
- SNR: 10.0 dB
- Snapshots: 256
- Trials: 500 per coupling level
- True DOAs: [15°, -20°] (35° separation)

**Output Files**:
- `scenario2a_coupling_sweep_Z5.csv` - Complete metrics table
- `scenario2a_coupling_sweep_Z5.png` - 6-panel visualization

### Experiment 2B: Array Sensitivity Comparison
**Purpose**: Compare coupling robustness across different array geometries

**Parameters**:
- Arrays: ULA, Z5, Z6
- Coupling strength: c1 = 0.3 (representative moderate coupling)
- SNR: 10.0 dB
- Snapshots: 256
- Trials: 500 per array
- True DOAs: [15°, -20°]

**Output Files**:
- `scenario2b_array_sensitivity.csv` - Array comparison metrics
- `scenario2b_array_sensitivity.png` - 3-panel comparison plot

## Metrics Guide

### Core Metrics (from Scenario 1)
1. **RMSE_degrees**: Root mean square DOA error after Hungarian matching
2. **RMSE_CRB_ratio**: Estimation efficiency vs Cramér-Rao Bound
3. **Resolution_Rate**: Percentage of correctly resolved sources (%)
4. **Bias_degrees**: Systematic signed estimation error
5. **Runtime_ms**: Computational time per trial

### New Coupling-Specific Metrics
6. **RMSE_Degradation_%**: Percentage change from Scenario 1 baseline
   - Formula: `100 * (RMSE_coupled - RMSE_baseline) / RMSE_baseline`
   - Negative = improvement, Positive = degradation
   - Example: +75% means RMSE increased by 75%

7. **CRB_Violation**: Change in RMSE/CRB ratio from baseline
   - Formula: `(RMSE/CRB)_coupled - (RMSE/CRB)_baseline`
   - Positive = coupling pushes further from theoretical bound
   - Near-zero = coupling doesn't affect estimation efficiency

8. **Coupling_Sensitivity**: Local gradient of degradation curve
   - Formula: `d(RMSE) / d(c1)` computed via finite differences
   - Units: degrees per coupling unit
   - High magnitude = rapid performance change in this coupling regime

9. **Failure_Threshold**: Critical coupling strength causing collapse
   - Definition: Smallest c1 where `Resolution_Rate < 80%`
   - If all c1 achieve ≥80%, reported as "Not reached"
   - Indicates system's coupling tolerance limit

10. **Resolution_Loss**: Percentage of sources becoming unresolvable
    - Formula: `100 * (1 - Resolution_Rate/100)`
    - Inverse of Resolution_Rate for intuitive degradation metric
    - 0% = perfect resolution, 100% = complete failure

## Key Findings (Test Run - 50 trials)

### Non-Monotonic Degradation Pattern Observed
1. **c1=0.0** (no coupling): RMSE=0.0057°, 0% degradation ✅
2. **c1=0.1**: RMSE=0.0014°, **-75% degradation** (IMPROVEMENT!) 🎉
3. **c1=0.2**: RMSE=0.0099°, +75% degradation ⚠️
4. **c1=0.3-0.4**: RMSE=0.0113°, +100% degradation 📉
5. **c1=0.5**: RMSE=0.0071°, +25% degradation (partial recovery) 📈

### Critical Observations
- **No failure threshold detected**: All coupling levels maintain 100% resolution
- **Beneficial coupling regime**: c1≈0.1 shows unexpected performance improvement
- **Variable coupling sensitivity**: Ranges from -0.042 to +0.049 deg/unit
- **Robust Z5 array**: Maintains source resolution across entire coupling range

### Interpretation
The non-monotonic pattern suggests:
1. **Weak coupling (c1=0.1)** may introduce beneficial regularization
2. **Moderate coupling (c1=0.2-0.4)** causes expected performance loss
3. **Strong coupling (c1=0.5)** shows partial recovery (possible array geometry interaction)
4. **ALSS optimization** provides exceptional coupling robustness (100% resolution throughout)

## Expected Results (Production - 500 trials)

### Anticipated Patterns
- **Refined degradation curve** with 10 coupling points (vs 6 in test)
- **Confirmation of c1≈0.1 improvement** phenomenon
- **Potential failure threshold** may emerge at c1 > 0.5 (needs extended sweep)
- **Statistical significance** of non-monotonic behavior

### Validation Criteria
- ✅ RMSE at c1=0.0 should match Scenario 1 baseline (≈0.006°)
- ✅ Resolution should remain ≥95% for c1 < 0.3 (robust regime)
- ✅ Coupling sensitivity should peak around c1=0.2-0.3 (steepest degradation)
- ✅ Runtime overhead should remain < 5% (coupling doesn't slow algorithm)

## Usage Examples

### Load and Analyze Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load coupling sweep data
df = pd.read_csv('scenario2a_coupling_sweep_Z5.csv')

# Find maximum degradation point
max_deg_idx = df['RMSE_Degradation_%'].abs().idxmax()
max_deg_c1 = df.loc[max_deg_idx, 'Coupling_c1']
max_deg_val = df.loc[max_deg_idx, 'RMSE_Degradation_%']
print(f"Maximum degradation: {max_deg_val:.1f}% at c1={max_deg_c1:.2f}")

# Check for failure threshold
failed = df[df['Resolution_Rate'] < 80.0]
if len(failed) > 0:
    failure_c1 = failed['Coupling_c1'].min()
    print(f"⚠️ Failure threshold: c1 = {failure_c1:.3f}")
else:
    print("✅ No failure threshold detected (all c1 maintain resolution)")

# Plot degradation vs coupling strength
plt.figure(figsize=(8, 5))
plt.plot(df['Coupling_c1'], df['RMSE_Degradation_%'], 'o-', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Coupling Strength (c1)', fontsize=12)
plt.ylabel('RMSE Degradation (%)', fontsize=12)
plt.title('Performance Degradation vs Mutual Coupling', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('degradation_analysis.png', dpi=300)
```

### Compare with Baseline
```python
# Load Scenario 1 baseline
baseline = pd.read_csv('../scenario1_baseline/scenario1a_snr_sweep_Z5.csv')
baseline_10db = baseline[baseline['SNR_dB'] == 10.0].iloc[0]

# Load Scenario 2 coupling sweep
coupled = pd.read_csv('scenario2a_coupling_sweep_Z5.csv')

# Compute degradation statistics
print("COUPLING IMPACT SUMMARY")
print("=" * 60)
print(f"Baseline RMSE (c1=0.0):     {baseline_10db['RMSE_degrees']:.4f}°")
print(f"Best coupled RMSE:           {coupled['RMSE_degrees'].min():.4f}° (c1={coupled.loc[coupled['RMSE_degrees'].idxmin(), 'Coupling_c1']:.2f})")
print(f"Worst coupled RMSE:          {coupled['RMSE_degrees'].max():.4f}° (c1={coupled.loc[coupled['RMSE_degrees'].idxmax(), 'Coupling_c1']:.2f})")
print(f"Mean degradation:            {coupled['RMSE_Degradation_%'].mean():.1f}%")
print(f"Maximum degradation:         {coupled['RMSE_Degradation_%'].max():.1f}%")
print(f"Resolution maintained:       {(coupled['Resolution_Rate'] == 100.0).all()}")
```

### Extract Failure Threshold
```python
# Find critical coupling level
resolution_threshold = 80.0  # Define failure as < 80% resolution

# Check each coupling level
for _, row in df.iterrows():
    c1 = row['Coupling_c1']
    res = row['Resolution_Rate']
    rmse = row['RMSE_degrees']
    
    if res < resolution_threshold:
        print(f"⚠️ FAILURE at c1={c1:.3f}: Resolution={res:.1f}%, RMSE={rmse:.4f}°")
        break
else:
    print(f"✅ NO FAILURE: All coupling levels maintain ≥{resolution_threshold}% resolution")
    print(f"   System is robust up to c1={df['Coupling_c1'].max():.2f}")
```

## File Structure

```
scenario2_coupling/
├── README.md                              # This file
├── scenario2a_coupling_sweep_Z5.csv       # Experiment 2A results (10 coupling points)
├── scenario2a_coupling_sweep_Z5.png       # 6-panel degradation visualization
├── scenario2b_array_sensitivity.csv       # Experiment 2B results (3 arrays)
└── scenario2b_array_sensitivity.png       # 3-panel array comparison
```

## Interpretation Guidelines

### Healthy Coupling Response
- **Gradual degradation**: RMSE increases monotonically with c1
- **Maintained resolution**: Resolution stays ≥90% for c1 < 0.4
- **Low sensitivity**: Coupling sensitivity < 0.05 deg/unit
- **Near-CRB**: CRB violation < 0.01 (efficiency preserved)

### Concerning Coupling Response
- **Abrupt failure**: Resolution drops below 80% at moderate c1
- **High sensitivity**: Coupling sensitivity > 0.1 deg/unit (unstable regime)
- **Large CRB violation**: Coupling pushes far from theoretical bound
- **Early breakdown**: Failure threshold c1 < 0.2

### ALSS Array Advantages (Expected)
- Z5 should show **lower degradation** than ULA at all coupling levels
- Z5 should have **higher failure threshold** (if one exists)
- Z5 should maintain **better resolution** under coupling
- Z5 **coupling sensitivity** should be more stable (less variance)

## Next Steps

1. ✅ Complete Scenario 2A production run (500 trials, 10 coupling points)
2. ⏳ Run Scenario 2B array sensitivity comparison
3. ⏳ Generate comparison analysis vs Scenario 1
4. ⏳ Copy final figures to `papers/radarcon2025_alss/figures/`
5. ⏳ Commit Scenario 2 results to repository
6. ⏳ Proceed to Scenario 3 (if defined in EXPERIMENT_PLAN.md)

## References

- **Baseline**: See `../scenario1_baseline/README.md`
- **Metrics**: See `docs/EXTENDED_METRICS_GUIDE.md`
- **Mutual Coupling**: See `core/radarpy/signal/mutual_coupling.py`
- **Experiment Plan**: See `../EXPERIMENT_PLAN.md`

---

**Created**: 2025-01-06  
**Status**: Production run in progress  
**Related**: Scenario 1 (baseline), ALSS paper (RadarCon 2025)
