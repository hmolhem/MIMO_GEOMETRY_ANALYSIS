# Extended Metrics Guide

## Overview

This guide covers the **extended metrics** beyond the core 5 metrics, including statistical validation and practical assessment metrics.

**Date:** November 6, 2025  
**Author:** MIMO Geometry Analysis Team

---

## Complete Metrics Suite (13 Total)

### Core Metrics (5)
1. **RMSE_degrees** - Root Mean Square Error
2. **RMSE_CRB_ratio** - Efficiency vs. Cramér-Rao Bound
3. **Resolution_Rate** - Success percentage
4. **Bias_degrees** - Systematic error
5. **Runtime_ms** - Computational efficiency

### Statistical Metrics (4)
6. **95%_Confidence_Intervals** - Uncertainty quantification
7. **Effect_Size** - Cohen's d for comparison
8. **Statistical_Power** - Detection probability
9. **Bootstrap_Validation** - Non-parametric verification

### Practical Metrics (4)
10. **Computational_Overhead_ms** - Additional runtime cost
11. **Memory_Footprint_MB** - RAM usage estimate
12. **Parameter_Sensitivity_Score** - Robustness measure
13. **Ease_of_Integration** - Usability assessment

---

## Statistical Metrics Details

### 1. 95% Confidence Intervals

**Purpose:** Quantify uncertainty in DOA estimates

**Method:** t-distribution based intervals for each DOA

**Output:**
```python
{
    'DOA_0': (lower_bound, upper_bound),  # In degrees
    'DOA_1': (lower_bound, upper_bound),
    ...
}
```

**Interpretation:**
- Narrower intervals → Higher precision
- Intervals crossing zero → Unbiased estimation
- Non-overlapping intervals → Statistically distinct DOAs

**Usage:**
```python
ci = compute_confidence_intervals(trials, true_doas, confidence_level=0.95)
print(f"DOA 0: [{ci['DOA_0'][0]:.2f}°, {ci['DOA_0'][1]:.2f}°]")
```

### 2. Effect Size (Cohen's d)

**Purpose:** Measure standardized difference between conditions

**Formula:** $d = \frac{\mu_{treatment} - \mu_{baseline}}{\sigma_{pooled}}$

**Interpretation:**
| |d| Value | Interpretation | Example |
|-----------|----------------|---------|
| < 0.2 | Negligible | Minor coupling effect |
| 0.2 - 0.5 | Small | Noticeable degradation |
| 0.5 - 0.8 | Medium | Significant impact |
| ≥ 0.8 | Large | Severe degradation |

**Usage:**
```python
effect_size = compute_effect_size(
    baseline_rmse=0.5,    # No coupling
    treatment_rmse=1.2,   # With coupling
    pooled_std=None       # Auto-computed
)
print(f"Effect size: {effect_size:.3f}")
# Output: Effect size: 1.400 (Large effect)
```

**Sign Convention:**
- **Positive:** Treatment worse than baseline (degradation)
- **Negative:** Treatment better than baseline (improvement)

### 3. Statistical Power

**Purpose:** Probability of correctly detecting an effect

**Formula:** Power = P(reject H₀ | H₁ is true)

**Interpretation:**
| Power Range | Interpretation | Recommendation |
|-------------|----------------|----------------|
| < 0.5 | Underpowered | Increase trial count |
| 0.5 - 0.8 | Moderate | Acceptable for exploratory |
| ≥ 0.8 | Good | Recommended minimum |
| ≥ 0.95 | Excellent | High confidence |

**Usage:**
```python
power = compute_statistical_power(
    effect_size=0.5,   # Medium effect
    n_trials=100,      # Sample size
    alpha=0.05         # Significance level
)
print(f"Statistical power: {power:.2%}")
# Output: Statistical power: 92.3%
```

**Application:** Determine required trial count:
```python
# Find required N for 80% power
for n in range(50, 500, 50):
    power = compute_statistical_power(0.5, n, 0.05)
    if power >= 0.80:
        print(f"Required trials: {n}")
        break
```

### 4. Bootstrap Validation

**Purpose:** Non-parametric confidence interval estimation

**Method:** Resampling with replacement (1000 bootstrap samples)

**Output:**
```python
{
    'RMSE': (lower_95, upper_95),
    'Bias': (lower_95, upper_95),
    'Resolution_Rate': (lower_95, upper_95)
}
```

**Advantages:**
- No distributional assumptions
- Robust to outliers
- Captures complex uncertainty

**Usage:**
```python
boot_ci = bootstrap_validation(
    estimated_trials, true_doas,
    n_bootstrap=1000,
    confidence_level=0.95
)
print(f"RMSE 95% CI: [{boot_ci['RMSE'][0]:.3f}, {boot_ci['RMSE'][1]:.3f}]")
```

**Interpretation:**
- Width indicates uncertainty
- Compare parametric vs. bootstrap CIs for validation
- Asymmetric intervals suggest skewed distribution

---

## Practical Metrics Details

### 1. Computational Overhead

**Purpose:** Measure additional runtime cost

**Metrics:**
- **Absolute Overhead:** Difference in milliseconds
- **Relative Overhead:** Percentage increase
- **Speedup Factor:** Ratio of runtimes

**Usage:**
```python
overhead = compute_computational_overhead(
    baseline_runtime_ms=10.0,    # Ideal array
    treatment_runtime_ms=15.0    # With MCM
)
print(f"Absolute: {overhead['Absolute_Overhead_ms']:.2f} ms")
print(f"Relative: {overhead['Relative_Overhead_percent']:.1f}%")
print(f"Speedup: {overhead['Speedup_Factor']:.2f}x")
```

**Interpretation:**
| Relative Overhead | Assessment | Action |
|-------------------|------------|--------|
| < 10% | Negligible | Acceptable |
| 10% - 50% | Moderate | Consider optimization |
| 50% - 200% | Significant | Profile bottlenecks |
| > 200% | Severe | Redesign required |

### 2. Memory Footprint

**Purpose:** Estimate RAM usage

**Components:**
- **Snapshot Matrix:** X (N × M) complex
- **Covariance Matrix:** Rxx (N × N) complex
- **Steering Matrix:** A (N × K) complex
- **Overhead Factor:** 1.5× for temporaries

**Usage:**
```python
memory = estimate_memory_footprint(
    array_size=7,      # N sensors
    snapshots=256,     # M samples
    num_sources=2,     # K sources
    dtype_bytes=16     # complex128
)
print(f"Total memory: {memory['Total_Estimated_MB']:.2f} MB")
```

**Typical Footprints:**
| Configuration | Memory | Notes |
|---------------|--------|-------|
| N=7, M=256, K=2 | 0.04 MB | Small array, typical |
| N=20, M=512, K=5 | 0.32 MB | Large array |
| N=50, M=1024, K=10 | 7.68 MB | Very large |

**Scaling:**
- Linear in M (snapshots)
- Quadratic in N (array size)
- Linear in K (sources)

### 3. Parameter Sensitivity Score

**Purpose:** Measure robustness to parameter variations

**Method:** Compute normalized performance variation across parameter sweep

**Formula:** 
$$\text{Sensitivity} = \frac{1}{P}\sum_{p=1}^P \frac{|\text{Metric}(p) - \text{Metric}_{baseline}|}{\text{Metric}_{baseline}}$$

**Interpretation:**
| Score | Robustness | Tuning Required |
|-------|------------|-----------------|
| < 0.1 | High | Minimal |
| 0.1 - 0.5 | Moderate | Careful selection |
| > 0.5 | Low | Extensive tuning |

**Usage:**
```python
# Sweep coupling strength: [0.1, 0.2, 0.3, 0.4, 0.5]
base_metrics = run_scenario(coupling_strength=0.3)  # Nominal
varied_metrics = [run_scenario(c) for c in [0.1, 0.2, 0.4, 0.5]]

sensitivity = compute_parameter_sensitivity(
    base_metrics, varied_metrics, [0.1, 0.2, 0.4, 0.5]
)
print(f"Overall sensitivity: {sensitivity['Overall_Sensitivity']:.3f}")
```

### 4. Ease of Integration

**Purpose:** Qualitative usability assessment

**Factors:**
- **API Complexity:** Low / Medium / High
- **Dependencies:** Required packages
- **Documentation:** Completeness
- **Integration Score:** 0 to 1 scale

**Scoring:**
```python
ease_of_integration = {
    'API_Complexity': 'Low',           # Simple function calls
    'Dependencies': 'NumPy, SciPy',    # Minimal requirements
    'Documentation': 'Complete',       # Full guide available
    'Integration_Score': 0.9           # Excellent (0-1)
}
```

---

## Using Extended Metrics

### Complete Analysis Workflow

```python
from core.radarpy.signal.metrics import compute_extended_metrics

# Run trials
trials = []
runtimes = []
for _ in range(500):
    start = time.perf_counter()
    est = run_music(...)
    runtime = (time.perf_counter() - start) * 1000
    trials.append(est)
    runtimes.append(runtime)

# Run baseline for comparison
baseline_trials = []
baseline_runtimes = []
for _ in range(500):
    start = time.perf_counter()
    est = run_music(..., coupling_matrix=None)  # No coupling
    runtime = (time.perf_counter() - start) * 1000
    baseline_trials.append(est)
    baseline_runtimes.append(runtime)

baseline_metrics = compute_scenario_metrics(
    baseline_trials, true_doas, positions, wavelength,
    snr_db, snapshots, None, 3.0, baseline_runtimes
)

# Compute extended metrics
extended = compute_extended_metrics(
    trials, true_doas, positions, wavelength,
    snr_db, snapshots, coupling_matrix,
    resolution_threshold=3.0,
    runtimes_ms=runtimes,
    baseline_metrics=baseline_metrics,  # For comparison
    n_bootstrap=1000
)

# Display comprehensive results
print_extended_metrics_summary(extended, "My Scenario")
```

### Example Output

```
======================================================================
  My Scenario
======================================================================

  CORE METRICS:
    RMSE:                  1.2345°
    RMSE/CRB Ratio:          2.34x
    Resolution Rate:        85.0%
    Bias:                  0.1234°
    Runtime:                12.34 ms

  STATISTICAL METRICS:
    RMSE 95% CI:         [1.1500, 1.3200]°
    Effect Size:            0.750 (Medium)
    Statistical Power:       98.5% (Excellent)
    Bootstrap Samples:       1000

  PRACTICAL METRICS:
    Memory Footprint:        0.04 MB
    Comp. Overhead:          23.4% increase
    Integration Score:        0.9/1.0

  CONFIGURATION:
    Trials:                   500
    Sensors:                    7
    Sources:                    2
    Snapshots:                256
======================================================================
```

---

## Statistical Considerations

### Sample Size Requirements

| Metric | Minimum Trials | Recommended | Notes |
|--------|---------------|-------------|-------|
| Core Metrics | 100 | 500 | Basic stability |
| Confidence Intervals | 30 | 100 | CLT assumption |
| Effect Size | 30 per group | 100 per group | Power analysis |
| Bootstrap | 100 | 500 | Resampling stability |

### Multiple Comparison Correction

When comparing multiple scenarios, apply Bonferroni correction:

```python
# For 5 comparisons, use α = 0.05 / 5 = 0.01
alpha_corrected = 0.05 / n_comparisons
power = compute_statistical_power(effect_size, n_trials, alpha_corrected)
```

### Reporting Standards

For publication, report:
1. **Core metrics** with 95% CIs
2. **Effect sizes** with confidence intervals
3. **Statistical power** achieved
4. **Bootstrap validation** results
5. **Sample sizes** used

**Example:**
> The mutual coupling matrix increased RMSE from 0.50° (95% CI: [0.45, 0.55]) to 1.20° (95% CI: [1.10, 1.30]), representing a large effect size (d = 1.40, 95% CI: [1.20, 1.60]). With 500 trials per condition, statistical power exceeded 99%.

---

## Integration with Experimental Scenarios

### Scenario 1: Baseline Performance
```python
extended = compute_extended_metrics(
    trials, true_doas, positions, wavelength, snr_db, snapshots,
    coupling_matrix=None,  # Ideal array
    baseline_metrics=None  # This IS the baseline
)
```

### Scenario 2: MCM Impact Study
```python
extended_coupled = compute_extended_metrics(
    trials, true_doas, positions, wavelength, snr_db, snapshots,
    coupling_matrix=C,
    baseline_metrics=baseline_extended  # Compare to Scenario 1
)
```

### Scenario 3: Parameter Sensitivity
```python
coupling_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
results = []
for strength in coupling_strengths:
    C = generate_mcm(N, strength, model='exponential')
    extended = compute_extended_metrics(trials, ..., coupling_matrix=C)
    results.append(extended)

# Analyze sensitivity
sensitivity = compute_parameter_sensitivity(
    results[2],  # Base = 0.3
    results,
    coupling_strengths
)
```

---

## API Reference

### Main Function

```python
def compute_extended_metrics(
    estimated_doas_trials: List[np.ndarray],
    true_doas: np.ndarray,
    sensor_positions: np.ndarray,
    wavelength: float,
    snr_db: float,
    snapshots: int,
    coupling_matrix: Optional[np.ndarray] = None,
    resolution_threshold: float = 3.0,
    runtimes_ms: Optional[List[float]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
    n_bootstrap: int = 1000
) -> Dict[str, any]:
```

### Individual Functions

```python
# Statistical metrics
compute_confidence_intervals(trials, true, confidence=0.95)
compute_effect_size(baseline_rmse, treatment_rmse, pooled_std=None)
compute_statistical_power(effect_size, n_trials, alpha=0.05)
bootstrap_validation(trials, true, n_bootstrap=1000, confidence=0.95)

# Practical metrics
compute_computational_overhead(baseline_ms, treatment_ms)
estimate_memory_footprint(N, M, K, dtype_bytes=16)
compute_parameter_sensitivity(base, varied_list, param_values)
```

---

## References

### Statistical Methods
- **Cohen's d:** Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences"
- **Bootstrap:** Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the Bootstrap"
- **Power Analysis:** Faul et al. (2007). "G*Power 3"

### DOA Estimation
- **CRB Theory:** Stoica & Nehorai (1989). IEEE Trans. ASSP
- **MUSIC Algorithm:** Schmidt, R. (1986). IEEE Trans. ASSP

---

**Status:** ✅ Extended metrics (13 total) ready for comprehensive evaluation!  
**Next Steps:** Apply to experimental scenarios with statistical rigor
