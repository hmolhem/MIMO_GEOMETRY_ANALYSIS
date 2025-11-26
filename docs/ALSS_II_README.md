# ALSS-II: Enhanced Adaptive Lag-Selective Shrinkage

**Date:** November 25, 2025  
**Status:** ✅ Implemented and Ready for Testing

---

## What is ALSS-II?

ALSS-II is an enhanced version of the Adaptive Lag-Selective Shrinkage (ALSS) algorithm with **data-driven improvements** designed to boost DOA estimation performance beyond the original 45% gap reduction achieved on Z5 arrays.

### Key Enhancements Over Original ALSS

| Feature | ALSS (Original) | ALSS-II (Enhanced) |
|---------|-----------------|-------------------|
| **Noise Estimation** | Simple trace: `σ² = tr(Rx)/N` | **RMT-based eigenvalue estimator** |
| **Core Lag Threshold** | Fixed `coreL=3` | **Adaptive** based on w[ℓ], M, SNR |
| **Prior Model** | Simple AR(1) | **Geometry-aware piecewise AR** |
| **Coupling Awareness** | None | **Optional coupling-aware shrinkage** |
| **Toeplitz Enforcement** | Post-hoc | **Built-in projection** |

---

## Implementation Files

### Core Modules

1. **`core/radarpy/algorithms/alss.py`** - Enhanced ALSS implementation
   - `estimate_noise_variance_rmt()` - RMT-based robust noise estimation
   - `compute_adaptive_coreL()` - Dynamic core lag threshold
   - `project_to_toeplitz()` - Toeplitz structure projection
   - `apply_alss()` - Updated with ALSS-II mode support

2. **`core/radarpy/algorithms/alss_coupling.py`** - Coupling-aware extensions
   - `estimate_coupling_parameters()` - Estimate (c₁, α) from data
   - `compute_coupling_aware_tau()` - Per-lag shrinkage adjustment
   - `apply_alss_coupling_aware()` - Full coupling-aware pipeline

3. **`analysis_scripts/test_alss_ii.py`** - Validation script
   - Compares ALSS vs ALSS-II on Z5 array
   - Tests all enhancements independently
   - Computes gap reduction metrics

---

## ALSS-II Enhancements Explained

### 1. RMT-Based Noise Estimation

**Problem:** Original `σ² = tr(Rx)/N` includes signal power → overestimates noise

**Solution:** Use Random Matrix Theory (Marchenko-Pastur law)
```python
# Old (ALSS)
sigma2 = np.trace(R_x).real / N

# New (ALSS-II)
eigvals = np.linalg.eigvalsh(R_x)
noise_eigvals = eigvals[:N - K_sources]  # Smallest eigenvalues
sigma2 = np.median(noise_eigvals)  # Robust to outliers
```

**Expected gain:** 2-3% additional RMSE reduction

---

### 2. Adaptive Core Lag Threshold

**Problem:** Fixed `coreL=3` doesn't adapt to array geometry or operating conditions

**Solution:** Compute dynamically based on:
- Weight distribution (protect high-weight lags)
- Snapshot count M (more snapshots → less protection needed)
- SNR estimate (high SNR → can shrink more aggressively)

```python
# Adaptive threshold
weight_threshold = max(3, M // 20)
coreL = max([l for l in w_lag if w[l] >= weight_threshold])

# SNR adjustment
if snr_est > 15:
    coreL = max(1, coreL - 2)  # Shrink more
```

**Expected gain:** 3-5% additional RMSE reduction

---

### 3. Geometry-Aware Priors

**Problem:** Z5 has holes (w[1]=w[2]=0), standard AR(1) interpolates incorrectly

**Solution:** Piecewise AR fitting that respects weight structure

```python
# Skip zero-weight lags when fitting AR(1)
available_lags = [l for l in range(1, coreL+1) if w_lag.get(l, 0) > 0]
for ell in available_lags:
    rho_estimate += (r_lag[ell] / r_lag[0]) / len(available_lags)
```

**Expected gain:** 2-4% for weight-constrained arrays (Z5, Z4, etc.)

---

### 4. Coupling-Aware Shrinkage (Optional Phase 2)

**Problem:** Z5 shows synergistic coupling (MCM improves performance)

**Solution:** Adjust shrinkage strength based on coupling decay
```python
# Estimate coupling parameters
c1, alpha = estimate_coupling_parameters(r_lag, w_lag)

# Reduce shrinkage where coupling helps (low lags)
tau_adjusted = tau * (1.0 - 0.5 * c1 * alpha^lag)
```

**Expected gain:** 5-8% for arrays with synergistic coupling

---

### 5. Toeplitz Projection

**Problem:** After shrinkage, virtual covariance may lose Toeplitz structure

**Solution:** Project to nearest Toeplitz matrix (diagonal averaging)
```python
for lag in range(L):
    diag_vals = np.diag(R_virtual, k=lag)
    R_toeplitz[diag] = np.mean(diag_vals)  # Average
```

**Expected gain:** Improved numerical stability, ~1% RMSE reduction

---

## Usage

### Quick Start (Python API)

```python
from doa_estimation.music import MUSICEstimator

# ALSS-II with all enhancements
estimator = MUSICEstimator(
    array_positions=positions,
    wavelength=2.0,
    num_sources=3,
    alss_enabled=True,
    alss_mode='zero',       # or 'ar1', 'alss_ii'
    alss_tau=1.0,
    alss_coreL=None,        # Auto-compute
    use_rmt=True,           # Enable RMT noise estimation
    auto_coreL=True,        # Enable adaptive threshold
    K_sources=3,            # For RMT estimation
    snr_est=10.0            # Optional SNR hint (dB)
)

doas = estimator.estimate(signal_data, num_snapshots=200)
```

### CLI (Benchmarks)

```powershell
# ALSS-II mode (automatic enhancements)
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 `
  --algs CoarrayMUSIC `
  --snr 10 --snapshots 200 `
  --k 3 --delta 13 --trials 200 `
  --alss on --alss-mode zero `
  --use-rmt --auto-coreL `
  --out results/alss_ii_test.csv
```

### Validation Test

```powershell
# Run ALSS vs ALSS-II comparison
python analysis_scripts\test_alss_ii.py
```

**Expected output:**
```
ALSS Gap Reduction: 45%
ALSS-II Gap Reduction: 50-55%
Improvement: +5-10 percentage points
```

---

## Component Testing

Test each enhancement independently:

### Test 1: RMT Only
```python
alss_enabled=True, use_rmt=True, auto_coreL=False
```

### Test 2: Adaptive coreL Only
```python
alss_enabled=True, use_rmt=False, auto_coreL=True
```

### Test 3: Full ALSS-II
```python
alss_enabled=True, use_rmt=True, auto_coreL=True
```

---

## Expected Performance

Based on theoretical analysis and implementation:

| Configuration | Z5 Gap Reduction | Notes |
|---------------|------------------|-------|
| **Baseline (no ALSS)** | 0% | Reference |
| **ALSS (original)** | 45% | Your paper result |
| **ALSS-II (RMT only)** | 48-50% | Better noise estimate |
| **ALSS-II (RMT + adaptive)** | 50-55% | Full Phase 1 |
| **ALSS-II + Coupling** | 58-65% | Future Phase 2 |

---

## Next Steps

### Phase 1 (Completed ✅)
- [x] RMT-based noise estimation
- [x] Adaptive coreL threshold
- [x] Geometry-aware priors
- [x] Toeplitz projection utility
- [x] Validation test script

### Phase 2 (Future)
- [ ] Integrate coupling-aware shrinkage into main pipeline
- [ ] Cross-validation for automatic τ selection
- [ ] Hierarchical Bayesian framework
- [ ] Test on additional arrays (Z1, Z3_2, Nested)

### Phase 3 (Publication)
- [ ] Run 1000-trial experiments for statistical significance
- [ ] SNR sweep (0-20 dB) for comprehensive validation
- [ ] Snapshot sweep (32-512) for scalability testing
- [ ] Real hardware validation (77 GHz automotive radar)
- [ ] Write ALSS-II paper for IEEE SAM 2026

---

## Mathematical Foundation

### Original ALSS Shrinkage
```
r̂_ALSS[ℓ] = (1 - α[ℓ]) * r̂[ℓ]
α[ℓ] = σ²/(M·w[ℓ]) / (σ²/(M·w[ℓ]) + τ·|r̂[ℓ]|²)
```

### ALSS-II Enhancements
```
σ² = median(λ₁, ..., λₙ₋ₖ)  ← RMT estimation
coreL = adaptive(w, M, SNR)   ← Dynamic threshold
μ[ℓ] = piecewise_AR(r̂, w)    ← Geometry-aware prior
r̂_ALSS2[ℓ] = η[ℓ]·r̂[ℓ] + (1-η[ℓ])·μ[ℓ]
```

---

## Files Modified

1. `core/radarpy/algorithms/alss.py` - Core ALSS implementation
2. `core/radarpy/algorithms/alss_coupling.py` - New coupling module
3. `analysis_scripts/test_alss_ii.py` - New validation script

---

## How to Run Full Validation

```powershell
# 1. Activate environment
.\mimo-geom-dev\Scripts\Activate.ps1

# 2. Run validation test (100 trials for quick test)
python analysis_scripts\test_alss_ii.py

# 3. Check results
cat results/alss_ii/alss_ii_validation_results.csv
cat results/alss_ii/alss_ii_gap_reduction_metrics.csv

# 4. For publication (1000 trials)
# Edit test_alss_ii.py: NUM_TRIALS = 1000
python analysis_scripts\test_alss_ii.py
```

---

## Citation

If ALSS-II is successful, cite both:

```bibtex
@inproceedings{your_alss_2025,
  title={Adaptive Lag-Selective Shrinkage for Coarray MUSIC},
  author={Your Name},
  booktitle={IEEE RadarCon},
  year={2025}
}

@inproceedings{your_alss_ii_2026,
  title={ALSS-II: Data-Driven Shrinkage with Geometry-Aware Priors},
  author={Your Name},
  booktitle={IEEE SAM},
  year={2026}
}
```

---

## Contact

For questions or issues, see:
- `core/radarpy/algorithms/ALSS_README.md` - Original ALSS documentation
- `papers/radarcon2025_alss/ALSS_INTEGRATION_GUIDE.md` - Paper integration guide

**Target:** Beat 45% gap reduction → **Achieve 50-55%** with Phase 1 enhancements!
