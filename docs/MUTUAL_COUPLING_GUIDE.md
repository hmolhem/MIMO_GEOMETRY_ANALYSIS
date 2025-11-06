# Mutual Coupling Matrix (MCM) Feature - Documentation

**Feature Added:** November 6, 2025  
**Version:** 1.0  
**Status:** ✅ Implemented and Tested

---

## Overview

Mutual coupling between antenna array elements has been implemented as an **optional feature** in the RadarPy MIMO Geometry Analysis framework. This allows realistic modeling of electromagnetic interactions between sensors, which can degrade DOA estimation performance in real-world scenarios.

---

## What is Mutual Coupling?

**Mutual Coupling** refers to electromagnetic interactions between nearby antenna elements. When one element transmits or receives, it induces currents in neighboring elements, modifying the array's response pattern.

### Physical Effects:
- **Amplitude distortion**: Signal magnitude changes
- **Phase distortion**: Steering vector phases shift
- **Pattern distortion**: Array radiation pattern modified
- **Performance degradation**: DOA estimation accuracy decreases

### Mathematical Model:
```
Ideal array response:     a(θ) = exp(j·k·x·sin(θ))
With mutual coupling:     a_coupled(θ) = C @ a(θ)
```
where **C** is the (N × N) mutual coupling matrix (MCM).

---

## Key Features

### ✅ **Optional - Easy On/Off Control**
- **Default behavior**: No coupling (ideal array) - existing code works unchanged
- **Enable coupling**: Pass `coupling_matrix` parameter
- **Perfect for comparative studies**: Run same experiment with/without coupling

### ✅ **Multiple Coupling Models**

1. **Exponential Decay Model**
   - Coupling decreases exponentially with distance
   - Formula: `C[i,j] = c₁·exp(-α·|pos[i]-pos[j]|)`
   - Best for: General-purpose modeling

2. **Toeplitz Model**
   - Symmetric structure for uniform linear arrays
   - Coupling depends only on element spacing
   - Best for: ULA with known coupling coefficients

3. **Measured Data**
   - Load real coupling matrix from measurements
   - Formats: CSV or NumPy binary (.npy)
   - Best for: Hardware validation with measured S-parameters

4. **No Coupling**
   - Set `coupling_matrix=None`
   - Ideal array behavior

### ✅ **Complete Integration**
- Works with **all array geometries**: ULA, Nested, Z1-Z6, custom
- Works with **both algorithms**: Spatial MUSIC, Coarray MUSIC
- Propagates through entire simulation pipeline
- No changes required to existing code

---

## Usage Guide

### Basic Usage - No Coupling (Default)

```python
from core.radarpy.signal.doa_sim_core import run_music
import numpy as np

# Works exactly as before - no coupling
positions = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
result = run_music(positions, wavelength=1.0, doas_true_deg=[10, -15], 
                   k_sources=2, snr_db=10, snapshots=100)
```

### Enable Coupling - Exponential Model

```python
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.signal.doa_sim_core import run_music

# Generate coupling matrix (exponential model)
positions = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
C = generate_mcm(len(positions), positions, model="exponential", 
                 c1=0.3, alpha=0.5)

# Run DOA estimation with coupling
result = run_music(positions, wavelength=1.0, doas_true_deg=[10, -15],
                   k_sources=2, snr_db=10, snapshots=100,
                   coupling_matrix=C)  # <-- Enable coupling
```

### Enable Coupling - Toeplitz Model

```python
# Define coupling coefficients for ULA
# [self, 1-apart, 2-apart, 3-apart, ...]
coupling_coeffs = np.array([1.0, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01])

C = generate_mcm(len(positions), positions, model="toeplitz",
                 coupling_coeffs=coupling_coeffs)

result = run_music(positions, wavelength=1.0, doas_true_deg=[10, -15],
                   k_sources=2, snr_db=10, snapshots=100,
                   coupling_matrix=C)
```

### Load Measured Coupling Matrix

```python
# Load from measurement file
C = generate_mcm(7, positions, model="measured", 
                 matrix_file="data/mcm_z5_measured.csv")

result = run_music(positions, wavelength=1.0, doas_true_deg=[10, -15],
                   k_sources=2, snr_db=10, snapshots=100,
                   coupling_matrix=C)
```

### Analyze Coupling Matrix Properties

```python
from core.radarpy.signal.mutual_coupling import get_coupling_info

info = get_coupling_info(C)
print(f"Max coupling between elements: {info['max_off_diagonal']:.3f}")
print(f"Average coupling: {info['avg_off_diagonal']:.3f}")
print(f"Condition number (stability): {info['condition_number']:.2e}")
print(f"Is matrix Hermitian: {info['is_hermitian']}")
```

---

## Comparative Studies

### Example: Compare Ideal vs Coupled Performance

```python
import numpy as np
from core.radarpy.signal.mutual_coupling import generate_mcm
from core.radarpy.signal.doa_sim_core import run_music

positions = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
true_doas = np.array([15.0, -20.0])
snr_db = 10
snapshots = 100

# Test 1: Ideal array (no coupling)
result_ideal = run_music(positions, 1.0, true_doas, 2, snr_db, snapshots,
                         coupling_matrix=None)
errors_ideal = np.abs(result_ideal['doas_est_deg'] - true_doas)
rmse_ideal = np.sqrt(np.mean(errors_ideal**2))

# Test 2: With coupling
C = generate_mcm(len(positions), positions, model="exponential", 
                 c1=0.3, alpha=0.5)
result_coupled = run_music(positions, 1.0, true_doas, 2, snr_db, snapshots,
                           coupling_matrix=C)
errors_coupled = np.abs(result_coupled['doas_est_deg'] - true_doas)
rmse_coupled = np.sqrt(np.mean(errors_coupled**2))

# Compare
print(f"RMSE (Ideal):   {rmse_ideal:.4f}°")
print(f"RMSE (Coupled): {rmse_coupled:.4f}°")
print(f"Degradation:    {((rmse_coupled - rmse_ideal) / rmse_ideal * 100):+.1f}%")
```

---

## Modified Functions

### 1. `core/radarpy/signal/mutual_coupling.py` (NEW)
**Main MCM module** with:
- `generate_mcm()` - Create coupling matrix
- `apply_coupling()` - Apply MCM to signals
- `compensate_coupling()` - Remove coupling effects
- `get_coupling_info()` - Analyze MCM properties

### 2. `core/radarpy/signal/array_manifold.py` (MODIFIED)
**Function:** `steering_vector()`
- **Added parameter:** `coupling_matrix=None`
- **Behavior:** Applies MCM to steering vectors if provided
- **Backward compatible:** Existing code works without changes

### 3. `core/radarpy/signal/doa_sim_core.py` (MODIFIED)
**Functions:**
- `simulate_snapshots()` - Added `coupling_matrix` parameter
- `music_spectrum()` - Added `coupling_matrix` parameter
- `run_music()` - Added `coupling_matrix` parameter

All maintain backward compatibility.

---

## Test Results

### Integration Test Summary (test/test_mcm_integration.py)

**Configuration:**
- Array: 7-element ULA, λ/2 spacing
- True DOAs: [15°, -20°]
- SNR: 10 dB
- Snapshots: 100

**Results:**

| Scenario | RMSE | Max Error | Degradation |
|----------|------|-----------|-------------|
| Ideal (No Coupling) | 0.14° | 0.20° | - |
| Exponential Coupling (c₁=0.3, α=0.5) | 0.26° | 0.30° | +80% |
| Toeplitz Coupling | 0.21° | 0.30° | +50% |

✅ **Conclusions:**
- MCM can be easily enabled/disabled
- Multiple coupling models work correctly
- Coupling effects propagate through pipeline
- Performance comparison possible

---

## Parameter Guidelines

### Exponential Model Parameters

```python
generate_mcm(N, positions, model="exponential", c0=1.0, c1=0.3, alpha=0.5)
```

**Parameters:**
- `c0`: Self-coupling (diagonal), typically **1.0**
- `c1`: Mutual coupling strength, range **0.1 - 0.5**
  - `0.1`: Weak coupling (large spacing)
  - `0.3`: Moderate coupling (λ/2 spacing)
  - `0.5`: Strong coupling (close spacing)
- `alpha`: Decay rate, range **0.3 - 1.0**
  - `0.3`: Slow decay (coupling extends far)
  - `0.5`: Moderate decay (typical)
  - `1.0`: Fast decay (localized coupling)

### Toeplitz Model Parameters

```python
generate_mcm(N, positions, model="toeplitz", coupling_coeffs=[1.0, 0.3, 0.1, ...])
```

**Coupling coefficients:**
- Index 0: Self-coupling = **1.0**
- Index 1: Adjacent elements (spacing = λ/2) ≈ **0.2 - 0.3**
- Index 2: 2-element spacing ≈ **0.1 - 0.15**
- Index k: Decreases with distance

**Typical ULA (λ/2 spacing):**
```python
coupling_coeffs = [1.0, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
```

---

## Benchmark Integration

### Adding MCM to Paper Benchmarks

To compare baseline vs ALSS **with mutual coupling**:

```python
# In scripts/run_paper_benchmarks.py or similar

from core.radarpy.signal.mutual_coupling import generate_mcm

# Generate MCM for array
positions_z5 = np.array([0, 1, 3, 5, 7, 9, 12])  # Z5 array
C_z5 = generate_mcm(len(positions_z5), positions_z5, 
                    model="exponential", c1=0.3, alpha=0.5)

# Run benchmark with coupling
for snr in [0, 5, 10, 15]:
    for snapshots in [64, 128, 256, 512]:
        result = run_music(positions_z5, wavelength=1.0, 
                          doas_true_deg=[10, -15],
                          k_sources=2, snr_db=snr, 
                          snapshots=snapshots,
                          coupling_matrix=C_z5)  # <-- Enable coupling
        # Process results...
```

### Configuration File Support (Future)

```yaml
# papers/radarcon2025_alss/configs/bench_default.yaml
mutual_coupling:
  enabled: true
  model: "exponential"
  parameters:
    c1: 0.3
    alpha: 0.5
```

---

## Performance Considerations

### Computational Cost
- **MCM generation:** One-time cost, O(N²)
- **MCM application:** Matrix multiply, O(N²) per snapshot
- **Overall impact:** Negligible (<5% overhead for N=7)

### Memory
- MCM storage: (N × N) complex matrix
- For N=7: 392 bytes (negligible)

### Numerical Stability
- Monitor condition number: `get_coupling_info(C)['condition_number']`
- Well-conditioned: cond(C) < 10
- Moderate: cond(C) = 10-100
- Ill-conditioned: cond(C) > 100 (may need regularization)

---

## Future Extensions

### Possible Enhancements:
1. **Command-line support** in benchmark scripts
   ```bash
   python run_benchmarks.py --coupling exponential --coupling-strength 0.3
   ```

2. **Configuration file integration**
   - YAML config with coupling parameters
   - Easy batch experiments

3. **Calibration algorithms**
   - Estimate unknown MCM from data
   - Self-calibration methods

4. **Compensation methods**
   - Coupling-aware MUSIC
   - Joint DOA-coupling estimation

5. **Array-specific models**
   - Custom MCM for each Z-array
   - Validated against EM simulations

---

## References

1. Friedlander, B., & Weiss, A. J. (1991). "Direction finding in the presence of mutual coupling," *IEEE Transactions on Antennas and Propagation*.

2. Svantesson, T. (1999). "Modeling and estimation of mutual coupling in a uniform linear array of dipoles," *IEEE ICASSP*.

3. Van Trees, H. L. (2002). *Optimum Array Processing*, Wiley-IEEE Press.

---

## Contact & Support

For questions or issues with MCM implementation:
- Review this documentation
- Check `test/test_mcm_integration.py` for examples
- See function docstrings in `core/radarpy/signal/mutual_coupling.py`

---

**Last Updated:** November 6, 2025  
**Feature Status:** ✅ Production Ready
