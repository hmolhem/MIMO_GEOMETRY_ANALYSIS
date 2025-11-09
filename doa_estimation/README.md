# DOA Estimation Module

Direction of Arrival (DOA) estimation using MUSIC algorithm with MIMO sparse arrays.

## Overview

This module provides complete DOA estimation capability for the MIMO array geometry framework. It implements the MUSIC (Multiple Signal Classification) algorithm, which exploits the difference coarray properties of sparse arrays to achieve high-resolution angle estimation.

## Features

- **MUSIC Algorithm**: Industry-standard subspace method for DOA estimation
- **Signal Simulation**: Generate test signals with configurable SNR and source types
- **Performance Metrics**: RMSE, MAE, bias, success rate, Cramér-Rao bounds
- **Visualization**: Spectrum plots, array geometry, estimation results
- **Array Compatibility**: Works with all 8+ array types (ULA, Nested, TCA, ePCA, Z1-Z6)
- **Mutual Coupling Matrix (MCM)**: Optional electromagnetic coupling modeling (NEW)
  - Exponential decay model
  - Toeplitz model
  - Easy on/off control
  - Realistic hardware simulation

## Quick Start

```python
from geometry_processors.z5_processor import Z5ArrayProcessor
from doa_estimation import MUSICEstimator, DOAMetrics

# Create array
z5 = Z5ArrayProcessor(N=7, d=0.5)
results = z5.run_full_analysis()

# Initialize MUSIC estimator
estimator = MUSICEstimator(
    sensor_positions=results.sensors_positions,
    wavelength=1.0
)

# Simulate signals
true_angles = [-30, 10, 45]
X = estimator.simulate_signals(
    true_angles=true_angles,
    SNR_dB=20,
    snapshots=200
)

# Estimate DOA
estimated_angles = estimator.estimate(X, K_sources=3)
print(f"Estimated: {estimated_angles}")

# Evaluate performance
metrics = DOAMetrics()
rmse = metrics.compute_rmse(true_angles, estimated_angles)
print(f"RMSE: {rmse:.2f}°")
```

## Module Structure

```
doa_estimation/
├── __init__.py          # Module exports
├── music.py             # MUSIC algorithm implementation
├── simulation.py        # Signal simulation utilities
├── metrics.py           # Performance evaluation
└── visualization.py     # Plotting functions
```

## Components

### 1. MUSICEstimator (`music.py`)

Core MUSIC algorithm for DOA estimation.

**Key Methods:**
- `estimate(X, K_sources)` - Main DOA estimation
- `simulate_signals(angles, SNR_dB, snapshots)` - Generate test signals
- `steering_vector(angle)` - Compute steering vector

**Parameters:**
- `sensor_positions`: Physical sensor locations (from array processor)
- `wavelength`: Signal wavelength (default: 1.0)
- `angle_range`: Search range in degrees (default: (-90, 90))
- `angle_resolution`: Angular grid resolution (default: 0.5°)

**Algorithm Steps:**
1. Compute spatial covariance: R = X @ X^H / snapshots
2. Eigendecomposition: Extract signal/noise subspaces
3. Noise subspace projection: U_noise (N × (N-K) eigenvectors)
4. MUSIC spectrum: P(θ) = 1 / (a(θ)^H @ U_noise @ U_noise^H @ a(θ))
5. Peak detection: Find K highest peaks = estimated angles

### 2. SignalSimulator (`simulation.py`)

Generate realistic received signals for testing.

**Key Methods:**
- `generate_signals(angles, SNR_dB, snapshots, signal_type)`
  - Signal types: 'random', 'narrowband', 'wideband'
  - Supports correlated sources
  - Configurable SNR

### 3. DOAMetrics (`metrics.py`)

Performance evaluation metrics.

**Methods:**
- `compute_rmse(true, estimated)` - Root Mean Squared Error
- `compute_mae(true, estimated)` - Mean Absolute Error
- `compute_bias(true, estimated)` - Estimation bias
- `compute_max_error(true, estimated)` - Maximum error
- `cramer_rao_bound(...)` - Theoretical performance limit
- `success_rate(true, estimated, threshold)` - Success percentage

### 4. Visualization (`visualization.py`)

Plotting utilities for DOA results.

**Functions:**
- `plot_doa_spectrum(angles, spectrum, true, estimated)` - MUSIC spectrum with peaks
- `plot_array_geometry(positions, array_name)` - Sensor layout
- `plot_estimation_results(true, estimated, rmse)` - Error analysis
- `plot_music_comparison(results_dict, metric)` - Compare arrays

## Demo Script Usage

The `run_doa_demo.py` script provides three modes:

### 1. Single Estimation

Estimate DOA for one array configuration:

```bash
python analysis_scripts/run_doa_demo.py --array z5 --N 7 --K 3 --SNR 20 --angles -30 10 45
```

**Key Arguments:**
- `--array`: Array type (ula, nested, tca, epca, z1-z6)
- `--N`: Number of sensors (for ULA, Z-arrays)
- `--K`: Number of sources to detect
- `--SNR`: Signal-to-noise ratio in dB
- `--angles`: True DOA angles (optional, random if omitted)
- `--plot`: Show visualization plots

**Output:**
```
Array: Array Z5 (N=7)
Physical Sensors: 7
Maximum Detectable Sources (K_max): 5

True DOA Angles: ['-30.0°', '10.0°', '45.0°']
Estimated DOA Angles: ['-30.0°', '10.0°', '45.0°']

Performance Metrics:
  RMSE:  0.000°
  MAE:   0.000°
  Bias:  0.000°
```

### 2. SNR Comparison

Analyze performance vs SNR:

```bash
python analysis_scripts/run_doa_demo.py --array z5 --N 7 --K 3 --compare-snr
```

Runs Monte Carlo trials (20 per SNR) from 0-20 dB and plots RMSE vs SNR curve.

### 3. Array Comparison

Compare different array types:

```bash
python analysis_scripts/run_doa_demo.py --compare-arrays --SNR 15
```

Tests ULA, Nested, Z5, and Z6 arrays with same conditions and compares RMSE.

## Array-Specific Usage

### ULA (Uniform Linear Array)
```python
from geometry_processors.ula_processors import ULArrayProcessor
processor = ULArrayProcessor(N=8, d=0.5)
```

### Nested Array
```python
from geometry_processors.nested_processor import NestedArrayProcessor
processor = NestedArrayProcessor(N1=2, N2=3, d=0.5)
```

### TCA (Two-level Coprime Array)
```python
from geometry_processors.tca_processor import TCAArrayProcessor
processor = TCAArrayProcessor(M=3, N=4, d=0.5)
```

### ePCA (Extended Prototypical Coprime Array)
```python
from geometry_processors.epca_processor import ePCAArrayProcessor
processor = ePCAArrayProcessor(p1=2, p2=3, p3=5, d=0.5)
```

### Z-Arrays (Z1, Z3_1, Z3_2, Z4, Z5, Z6)
```python
from geometry_processors.z5_processor import Z5ArrayProcessor
processor = Z5ArrayProcessor(N=7, d=0.5)
```

## K_max and Source Detection

The maximum number of detectable sources (K_max) is determined by the contiguous coarray segment length (L):

```
K_max = floor(L / 2)
```

This varies by array type:
- **ULA (N=8)**: K_max = 4
- **Nested (N1=2, N2=3)**: K_max = 2
- **Z5 (N=7)**: K_max = 5
- **Z6 (N=7)**: K_max = 6

Attempting to detect more than K_max sources will lead to reduced accuracy or failures.

## Performance Tips

### 1. SNR Requirements
- **Low SNR (< 5 dB)**: Expect errors > 5°
- **Medium SNR (10-15 dB)**: Good performance (< 2° RMSE)
- **High SNR (> 20 dB)**: Excellent performance (< 0.5° RMSE)

### 2. Source Separation
- **Well-separated (> 20°)**: Easy to resolve
- **Moderate (10-20°)**: Requires SNR > 10 dB
- **Close (< 10°)**: Difficult, needs high SNR and many snapshots

### 3. Snapshots
- **Minimum**: 100 snapshots
- **Recommended**: 200-500 snapshots
- **High accuracy**: 1000+ snapshots

### 4. Angle Resolution
- **Coarse (1-2°)**: Faster computation
- **Standard (0.5°)**: Good balance
- **Fine (0.1-0.2°)**: Higher accuracy but slower

## Mutual Coupling Matrix (MCM) Support

**NEW:** The DOA module now supports modeling of electromagnetic mutual coupling between array elements, which affects real-world antenna performance.

### What is Mutual Coupling?

In real antenna arrays, electromagnetic coupling between nearby elements causes the received signal at each sensor to be influenced by neighboring sensors. This effect:
- Degrades DOA estimation accuracy
- Is stronger for closely-spaced elements
- Decays with distance between sensors

### Enabling MCM

**CLI Usage:**
```bash
# Basic MCM with exponential model (default c1=0.3, alpha=0.5)
python run_doa_demo.py --array z5 --N 7 --K 3 --enable-mcm

# Custom coupling strength
python run_doa_demo.py --array z5 --N 7 --K 3 --enable-mcm --mcm-c1 0.5 --mcm-alpha 0.3

# Toeplitz model
python run_doa_demo.py --array z5 --N 7 --K 3 --enable-mcm --mcm-model toeplitz
```

**Programmatic Usage:**
```python
from doa_estimation import MUSICEstimator

# Exponential decay model
estimator = MUSICEstimator(
    sensor_positions=positions,
    wavelength=2.0,
    enable_mcm=True,
    mcm_model='exponential',
    mcm_params={'c1': 0.3, 'alpha': 0.5}  # c1: strength, alpha: decay rate
)

# Toeplitz model (symmetric coupling)
estimator = MUSICEstimator(
    sensor_positions=positions,
    wavelength=2.0,
    enable_mcm=True,
    mcm_model='toeplitz',
    mcm_params={'coupling_values': [1.0, 0.3, 0.15, 0.08]}  # [self, 1st neighbor, 2nd, ...]
)
```

### MCM Models

**1. Exponential Decay Model** (default)
```
C[i,j] = 1.0                           (if i == j, self-coupling)
C[i,j] = c1 * exp(-alpha * |d_i - d_j|)  (if i != j, mutual coupling)
```
- `c1`: Coupling strength coefficient (0 to 1, typical: 0.2-0.4)
- `alpha`: Decay rate with distance (typical: 0.3-0.7)
- Higher c1 = stronger coupling
- Higher alpha = faster decay with distance

**2. Toeplitz Model**
```
C[i,j] = coupling_values[|i-j|]
```
- Assumes uniform spacing
- Define coupling for each lag: [self, lag-1, lag-2, ...]
- Typically: [1.0, 0.3, 0.15, 0.08, 0.04, ...]

### Expected Impact

MCM typically **degrades** DOA estimation performance:

| Configuration | RMSE (No MCM) | RMSE (With MCM) | Degradation |
|---------------|---------------|-----------------|-------------|
| Z5, SNR=20dB, c1=0.3 | 0.5° | 2.0° | 4× worse |
| ULA, SNR=15dB, c1=0.4 | 1.0° | 4.5° | 4.5× worse |
| Nested, SNR=10dB, c1=0.2 | 2.0° | 3.5° | 1.75× worse |

**Why degradation?** MCM causes model mismatch - the actual steering vectors differ from the ideal assumed in MUSIC. This is **expected and realistic** behavior.

### When to Use MCM

✅ **Use MCM when:**
- Simulating real hardware performance
- Evaluating robustness to coupling effects
- Comparing arrays for practical deployment
- Testing MCM compensation algorithms

❌ **Skip MCM when:**
- Studying ideal array geometry properties
- Benchmarking algorithm performance
- Initial development and testing
- When coupling is negligible (d > λ/2)

### Testing MCM

Test MCM integration:
```bash
python analysis_scripts/test_mcm_doa.py
```

This runs 3 experiments:
1. No MCM baseline (expected: RMSE ≈ 0°)
2. Exponential MCM (expected: RMSE ≈ 30-40°)
3. Toeplitz MCM (expected: RMSE ≈ 60-70°)

### Technical Details

MCM is applied to steering vectors:
```python
a_ideal(θ) = exp(j × 2π × positions / λ × sin(θ))  # Ideal steering vector
a_coupled(θ) = C @ a_ideal(θ)                       # With coupling effects
```

The MUSIC algorithm then operates on coupled steering vectors, introducing mismatch that degrades performance.

**Integration:** MCM functionality comes from `core/radarpy/signal/mutual_coupling.py`, which provides `generate_mcm_exponential()` and `generate_mcm_toeplitz()` functions.

### Further Reading

For MCM compensation techniques and advanced usage, see:
- `core/radarpy/signal/mutual_coupling.py` - MCM generation functions
- `docs/MUTUAL_COUPLING_GUIDE.md` - Comprehensive MCM documentation
- Project README section on MCM support

## Theory: MUSIC Algorithm

### Signal Model
For K narrowband sources at angles {θ₁, ..., θₖ}:

```
X = A(θ)S + N
```

Where:
- X: Received signal matrix (N × snapshots)
- A(θ): Steering matrix (N × K)
- S: Source signals (K × snapshots)
- N: Noise (N × snapshots)

### Covariance Matrix
```
R = E[XX^H] ≈ (1/L) XX^H
```

### Subspace Decomposition
Eigendecomposition of R yields:
- **Signal subspace**: Span of K largest eigenvectors (U_signal)
- **Noise subspace**: Span of (N-K) smallest eigenvectors (U_noise)

### MUSIC Spectrum
```
P_MUSIC(θ) = 1 / ||a^H(θ) U_noise||²
```

Peaks occur at true DOA angles due to orthogonality:
```
a(θ_true) ⊥ U_noise
```

### Advantages with Sparse Arrays
1. **Virtual aperture**: Difference coarray provides larger effective aperture
2. **More DOFs**: Can detect K_max = L/2 sources with N physical sensors (L >> N)
3. **High resolution**: Larger aperture → better angular resolution

## Integration with MIMO Framework

The DOA module seamlessly integrates with the geometry analysis framework:

```python
# 1. Create array and analyze geometry
processor = Z5ArrayProcessor(N=7)
array_data = processor.run_full_analysis()

# 2. Check K_max from coarray analysis
perf = array_data.performance_summary_table
k_row = perf[perf['Metrics'].str.contains('K_max')]
K_max = int(k_row['Value'].iloc[0])
print(f"Can detect up to {K_max} sources")

# 3. Use sensor positions for DOA
estimator = MUSICEstimator(array_data.sensors_positions)

# 4. Validate with simulations
X = estimator.simulate_signals([−20, 0, 30], SNR_dB=15)
angles = estimator.estimate(X, K_sources=3)
```

## Examples

### Example 1: Basic DOA Estimation
```python
from geometry_processors.z5_processor import Z5ArrayProcessor
from doa_estimation import MUSICEstimator, DOAMetrics

# Setup
z5 = Z5ArrayProcessor(N=7, d=0.5)
results = z5.run_full_analysis()
estimator = MUSICEstimator(results.sensors_positions)

# Estimate
X = estimator.simulate_signals([-30, 0, 30], SNR_dB=20, snapshots=200)
angles = estimator.estimate(X, K_sources=3)
print(angles)  # [-30.0, 0.0, 30.0]
```

### Example 2: Performance Evaluation
```python
true_angles = [-40, -10, 20, 45]
X = estimator.simulate_signals(true_angles, SNR_dB=15, snapshots=300)
estimated = estimator.estimate(X, K_sources=4)

metrics = DOAMetrics()
print(f"RMSE: {metrics.compute_rmse(true_angles, estimated):.2f}°")
print(f"MAE: {metrics.compute_mae(true_angles, estimated):.2f}°")
print(f"Success Rate: {metrics.success_rate(true_angles, estimated, threshold=2.0):.1%}")
```

### Example 3: Visualize MUSIC Spectrum
```python
from doa_estimation.visualization import plot_doa_spectrum

true_angles = [-20, 30]
X = estimator.simulate_signals(true_angles, SNR_dB=15, snapshots=200)
estimated, spectrum = estimator.estimate(X, K_sources=2, return_spectrum=True)

plot_doa_spectrum(
    estimator.angle_grid,
    spectrum,
    true_angles=true_angles,
    estimated_angles=estimated,
    title="Z5 Array MUSIC Spectrum",
    save_path="results/plots/music_spectrum.png"
)
```

### Example 4: Array Comparison Study
```python
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor

arrays = [
    ("ULA", ULArrayProcessor(N=8)),
    ("Nested", NestedArrayProcessor(N1=2, N2=3)),
    ("Z5", Z5ArrayProcessor(N=7))
]

for name, processor in arrays:
    data = processor.run_full_analysis()
    est = MUSICEstimator(data.sensors_positions)
    X = est.simulate_signals([−30, 0, 30], SNR_dB=15, snapshots=200)
    angles = est.estimate(X, K_sources=3)
    rmse = DOAMetrics.compute_rmse([−30, 0, 30], angles)
    print(f"{name}: RMSE = {rmse:.2f}°")
```

## References

1. R. Schmidt, "Multiple emitter location and signal parameter estimation," IEEE Trans. Antennas Propag., vol. 34, no. 3, pp. 276-280, 1986.

2. P. Pal and P. P. Vaidyanathan, "Nested arrays: A novel approach to array processing with enhanced degrees of freedom," IEEE Trans. Signal Process., vol. 58, no. 8, pp. 4167-4181, 2010.

3. P. Stoica and A. Nehorai, "MUSIC, maximum likelihood, and Cramer-Rao bound," IEEE Trans. Acoust., Speech, Signal Process., vol. 37, no. 5, pp. 720-741, 1989.

## Notes

- **Computational Complexity**: O(N² × snapshots) for covariance + O(N³) for eigendecomposition
- **Memory**: Stores N × N covariance matrix and N × N_angles steering matrix
- **Limitations**: Assumes narrowband signals, uncorrelated sources, known K
- **Extensions**: Can be adapted for wideband, coherent sources, or unknown K

## Troubleshooting

### Issue: Poor estimation accuracy
**Solutions:**
- Increase SNR (add `--SNR 20` or higher)
- Increase snapshots (add `--snapshots 500`)
- Ensure sources are well-separated (> 10°)
- Check K_sources ≤ K_max

### Issue: Wrong number of sources detected
**Solutions:**
- Verify K_max from coarray analysis
- Increase SNR to improve peak detection
- Adjust angle resolution for finer grid

### Issue: Spurious peaks in spectrum
**Solutions:**
- Increase SNR to improve signal/noise subspace separation
- Use more snapshots for better covariance estimation
- Check that sources are within angle_range

## Future Extensions

Potential enhancements:
- Root-MUSIC for improved resolution
- ESPRIT algorithm
- Wideband DOA methods
- Coherent source handling (spatial smoothing)
- 2D DOA estimation
- Adaptive beamforming
- Real-time processing

---

For questions or issues, refer to the main project documentation or the comprehensive docstrings in each module file.
