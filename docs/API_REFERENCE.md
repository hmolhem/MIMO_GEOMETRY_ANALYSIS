# API Reference

Complete reference for programmatic usage of the MIMO Array Geometry Analysis Framework.

---

## Core Classes

### `ArraySpec` (Data Container)

Located in: `geometry_processors/bases_classes.py`

**Purpose:** Structured data container with 47 pre-defined attributes for storing analysis results.

**Key Attributes by Phase:**

```python
# Physical Array (Input)
sensor_positions: np.ndarray    # Physical sensor positions
num_sensors: int                # Total sensor count (N)
sensor_spacing: float           # Base spacing multiplier (d)

# Coarray Data (Computed)
unique_differences: np.ndarray  # All unique difference values
coarray_positions: np.ndarray   # Virtual sensor positions
virtual_only_positions: np.ndarray  # Positions not in physical array
num_virtual: int                # Total virtual sensors (Mv)

# Weight Distribution (Analysis)
weight_table: pd.DataFrame      # Columns: ['Lag', 'Weight']
weights_dict: dict              # {lag: weight} mapping

# Segment Analysis (Performance)
contiguous_segments: List[Tuple[int, int, int]]  # (start, end, length)
max_segment_length: int         # Longest contiguous segment (L)
holes: List[int]                # Missing positions in [-max, max]

# Performance Metrics (Summary)
performance_summary_table: pd.DataFrame  # Final metrics table
max_detectable_sources: int     # K_max = floor(L/2)
coarray_aperture: int           # span(virtual_array)
```

**Usage Pattern:**

```python
from geometry_processors.z5_processor_ import Z5ArrayProcessor

processor = Z5ArrayProcessor(N=7, d=1.0)
spec = processor.run_full_analysis()

# Access attributes via .data
print(f"Virtual sensors: {spec.num_virtual}")
print(f"Max sources: {spec.max_detectable_sources}")
print(spec.performance_summary_table.to_markdown(index=False))
```

---

### `BaseArrayProcessor` (Abstract Framework)

Located in: `geometry_processors/bases_classes.py`

**Purpose:** Abstract base class defining the standardized 7-step analysis pipeline.

**Abstract Methods (must implement):**

```python
class BaseArrayProcessor(ABC):
    @abstractmethod
    def define_physical_array(self) -> None:
        """Step 1: Define sensor positions"""
        pass
    
    @abstractmethod
    def compute_all_differences(self) -> None:
        """Step 2: Calculate N² pairwise differences"""
        pass
    
    @abstractmethod
    def analyze_coarray(self) -> None:
        """Step 3: Identify unique positions, holes, segments"""
        pass
    
    @abstractmethod
    def compute_weight_distribution(self) -> None:
        """Step 4: Count frequency of each lag"""
        pass
    
    @abstractmethod
    def analyze_contiguous_segments(self) -> None:
        """Step 5: Find longest hole-free segments"""
        pass
    
    @abstractmethod
    def analyze_holes(self) -> None:
        """Step 6: Identify missing positions"""
        pass
    
    @abstractmethod
    def generate_performance_summary(self) -> None:
        """Step 7: Create metrics table"""
        pass
    
    @abstractmethod
    def plot_coarray(self) -> None:
        """Visualize virtual array"""
        pass
```

**Pipeline Method:**

```python
def run_full_analysis(self) -> ArraySpec:
    """
    Execute all 7 steps in sequence.
    
    Returns:
        ArraySpec: Complete analysis results with all attributes populated
    """
    self.define_physical_array()
    self.compute_all_differences()
    self.analyze_coarray()
    self.compute_weight_distribution()
    self.analyze_contiguous_segments()
    self.analyze_holes()
    self.generate_performance_summary()
    return self.data
```

---

## Array Processors

### ULA (Uniform Linear Array)

```python
from geometry_processors.ula_processors_ import ULArrayProcessor

# Create processor
ula = ULArrayProcessor(M=4, d=1.0)

# Parameters:
#   M (int): Number of sensors
#   d (float): Inter-element spacing multiplier

# Run analysis
results = ula.run_full_analysis()

# Expected output:
#   Physical: [0, 1, 2, 3] (M=4, d=1.0)
#   Virtual: [-3, -2, -1, 0, 1, 2, 3] (Mv=7)
#   K_max: 3 (can detect 3 sources)
```

### Nested Array

```python
from geometry_processors.nested_processor_ import NestedArrayProcessor

# Create processor
nested = NestedArrayProcessor(N1=2, N2=3, d=1.0)

# Parameters:
#   N1 (int): Inner array size
#   N2 (int): Outer array size
#   d (float): Base spacing

# Run analysis
results = nested.run_full_analysis()

# Structure:
#   Inner: [0, 1, 2, ..., N1-1]
#   Outer: [0, N1+1, 2(N1+1), ..., (N2-1)(N1+1)]
#   Total: N = N1 + N2 sensors
```

### Z5 Array (Specialized)

```python
from geometry_processors.z5_processor_ import Z5ArrayProcessor

# Create processor with canonical N=7 layout
z5 = Z5ArrayProcessor(N=7, d=1.0)

# Parameters:
#   N (int): Must be 7 for canonical layout [0,5,8,11,14,17,21]
#   d (float): Physical spacing multiplier

# Run analysis
results = z5.run_full_analysis()

# Properties:
#   Virtual sensors (Mv): 43
#   Max sources (K_max): 21
#   Weight at lag 1: 8
#   Weight at lag 2: 0 (key property)
#   Holes: 0 (fully contiguous)
```

### Z4 Array (Specialized)

```python
from geometry_processors.z4_processor_ import Z4ArrayProcessor

# Create with assertions for theoretical validation
z4 = Z4ArrayProcessor(N=7, d=1.0, do_asserts=True)

# Parameters:
#   N (int): Must be 7 for canonical layout [0,5,8,11,14,17,21]
#   d (float): Spacing multiplier
#   do_asserts (bool): Enable invariant checking

# Run analysis
results = z4.run_full_analysis()

# Assertions verify:
#   w(1) = w(2) = 0 (required by design)
#   L1 = 3 (first segment length)
```

---

## DOA Estimation

### Spatial MUSIC (Physical Array)

```python
from core.radarpy.algorithms.spatial_music import estimate_doa_spatial_music
from core.radarpy.signal.doa import generate_snapshot_matrix
import numpy as np

# Generate test data
sensor_positions = np.array([0, 5, 8, 11, 14, 17, 21])
true_doas = np.array([10.0, 23.0])  # Degrees
wavelength = 1.0
snapshots = 64
snr_db = 10

X = generate_snapshot_matrix(
    sensor_positions, true_doas, wavelength, snapshots, snr_db
)

# Compute sample covariance
Rxx = (X @ X.conj().T) / snapshots

# Run Spatial MUSIC
doas_est, info = estimate_doa_spatial_music(
    Rxx=Rxx,
    sensor_positions=sensor_positions,
    wavelength=wavelength,
    num_sources=2,
    search_grid=np.arange(-90, 90, 0.1)
)

# Results
print(f"Estimated DOAs: {doas_est}")
print(f"Condition number: {info['Rx_cond']:.2f}")
print(f"Singular values: {info['Rx_singular']}")
```

### Coarray MUSIC (Virtual Array)

```python
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music
from core.radarpy.signal.doa import generate_snapshot_matrix
import numpy as np

# Generate test data
sensor_positions = np.array([0, 5, 8, 11, 14, 17, 21])
true_doas = np.array([10.0, 23.0])
wavelength = 1.0
snapshots = 64
snr_db = 10

X = generate_snapshot_matrix(
    sensor_positions, true_doas, wavelength, snapshots, snr_db
)

# Run Coarray MUSIC
doas_est, info = estimate_doa_coarray_music(
    X=X,
    sensor_positions=sensor_positions,
    wavelength=wavelength,
    num_sources=2,
    search_grid=np.arange(-90, 90, 0.1)
)

# Results
print(f"Estimated DOAs: {doas_est}")
print(f"Virtual sensors: {info['Mv']}")
print(f"Condition number: {info['Rv_cond']:.2f}")
```

### Coarray MUSIC with ALSS

```python
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music

# Enable ALSS regularization
doas_est, info = estimate_doa_coarray_music(
    X=X,
    sensor_positions=sensor_positions,
    wavelength=wavelength,
    num_sources=2,
    search_grid=np.arange(-90, 90, 0.1),
    alss_enabled=True,
    alss_mode='zero',      # Shrink toward zero
    alss_tau=1.0,          # Shrinkage intensity
    alss_protected_core=3  # Protect first 3 lags
)

# Benefits:
#   - Improved conditioning (lower κ)
#   - Better RMSE for small snapshots
#   - Reduced estimation variance
```

---

## Metrics and Performance

### Cramér-Rao Bound (CRB)

```python
from core.radarpy.metrics.crb_doa import crb_pair_worst_deg

# Compute CRB for two-source scenario
crb_var = crb_pair_worst_deg(
    Mv=43,                      # Virtual aperture (e.g., Z5)
    d=1.0,                      # Spacing multiplier
    wavelength=1.0,             # Carrier wavelength
    snr_linear=10**(5/10),      # SNR = 5 dB
    theta_pair_deg=np.array([10.0, 23.0]),  # True DOAs
    snapshots=64                # Temporal snapshots
)

crb_deg = np.sqrt(crb_var)
print(f"CRB (std dev): {crb_deg:.4f}°")
```

### Resolve Tolerance Check

```python
from scripts.run_paper_benchmarks import resolve_tolerance_check

# Check if estimates satisfy tolerance criteria
is_resolved = resolve_tolerance_check(
    est_doas=np.array([9.8, 23.2]),       # Estimated DOAs
    true_doas=np.array([10.0, 23.0]),     # True DOAs
    pos_tol_deg=1.0,                      # ±1° position tolerance
    sep_tol_deg=0.5                       # ≥0.5° separation tolerance
)

# Returns True if:
#   1. len(est_doas) == len(true_doas)
#   2. All |est - true| ≤ 1.0°
#   3. |est[1] - est[0]| ≥ 0.5°
```

---

## Benchmarking Tools

### Run Single Trial

```python
from scripts.run_paper_benchmarks import run_single_trial
import numpy as np

# Setup
sensor_positions = np.array([0, 5, 8, 11, 14, 17, 21])
true_doas = np.array([10.0, 23.0])
wavelength = 1.0
snapshots = 64
snr_db = 10
coarse_grid = np.arange(-90, 90, 0.05)  # 0.05° coarse
fine_step = 0.01                         # 0.01° refinement

# Run trial
doas_est = run_single_trial(
    sensor_positions, true_doas, wavelength, snapshots, snr_db,
    coarse_grid, fine_step, num_sources=2,
    alss_enabled=True, alss_mode='zero', tau=1.0, ell_0=3
)

# Compute RMSE
rmse = np.sqrt(np.mean((doas_est - true_doas)**2))
print(f"RMSE: {rmse:.4f}°")
```

### Run Benchmark Sweep

```python
from scripts.run_paper_benchmarks import run_benchmark_sweep

# Configure sweep
config = {
    'array': 'Z5',
    'N': 7,
    'deltas': [10, 13, 20, 30, 45],
    'snr_vals': [0, 5, 10, 15, 20],
    'trials': 400,
    'snapshots': 64,
    'alss_mode': 'zero',
    'output_csv': 'results/bench/z5_sweep.csv'
}

# Run (takes ~4-6 hours for 25 scenarios × 400 trials)
run_benchmark_sweep(**config)
```

---

## Plotting Tools

### Plot RMSE with Confidence Intervals

```python
from tools.plot_paper_benchmarks import plot_rmse_with_ci
import pandas as pd

# Load benchmark results
df = pd.read_csv('results/bench/z5_paper_N7_T400_alss_zero.csv')

# Generate plot
plot_rmse_with_ci(
    df, 
    output_path='results/bench/figures/rmse_plot.png',
    title='Z5 Array: RMSE vs SNR'
)

# Creates:
#   - 6-panel subplot (2 rows × 3 columns)
#   - Bootstrap 95% CIs as error bars
#   - CRB overlay for comparison
#   - PNG (300 DPI) + PDF (vector)
```

### Plot Condition Numbers

```python
from tools.plot_paper_benchmarks import plot_condition_numbers

# Plot Rx vs Rv condition numbers
plot_condition_numbers(
    df,
    output_path='results/bench/figures/cond_plot.png'
)

# Shows:
#   - Semilogy plot (log scale for κ)
#   - Rx (physical) vs Rv (virtual)
#   - Separate lines for each delta value
```

---

## Utilities

### Weight Distribution Analysis

```python
from geometry_processors.z5_processor_ import Z5ArrayProcessor

processor = Z5ArrayProcessor(N=7, d=1.0)
results = processor.run_full_analysis()

# Access weight table
wt = results.weight_table
print(wt.to_markdown(index=False))

# Convert to dict for quick lookup
weights_dict = {int(row['Lag']): int(row['Weight']) 
                for _, row in wt.iterrows()}

print(f"Weight at lag 1: {weights_dict.get(1, 0)}")
print(f"Weight at lag 2: {weights_dict.get(2, 0)}")
```

### Contiguous Segment Extraction

```python
# Get longest contiguous segment
segments = results.contiguous_segments
longest_seg = max(segments, key=lambda x: x[2])

print(f"Longest segment: [{longest_seg[0]}, {longest_seg[1]}]")
print(f"Length: {longest_seg[2]}")
print(f"K_max: {longest_seg[2] // 2}")
```

### Holes Analysis

```python
# Identify missing positions
holes = results.holes
if holes:
    print(f"Found {len(holes)} holes: {holes}")
else:
    print("No holes (fully contiguous coarray)")
```

---

## Advanced Features

### Custom Array Design

```python
from geometry_processors.bases_classes import BaseArrayProcessor
import numpy as np

class CustomArrayProcessor(BaseArrayProcessor):
    def __init__(self, custom_positions):
        super().__init__(
            name="CustomArray",
            array_type="Custom",
            sensor_positions=np.array(custom_positions)
        )
    
    # Implement all 8 abstract methods...
    # See DEVELOPMENT_GUIDE.md for full template

# Usage
custom = CustomArrayProcessor([0, 3, 7, 15, 20])
results = custom.run_full_analysis()
```

### SVD Analysis

```python
# Run benchmark with SVD capture
import subprocess

cmd = [
    'python', 'core/analysis_scripts/run_benchmarks.py',
    '--arrays', 'Z5',
    '--N', '7',
    '--dump-svd',  # Enable SVD capture
    '--out', 'results/bench/test.csv'
]

subprocess.run(cmd, shell=True)

# Analyze singular values
subprocess.run([
    'python', 'tools/analyze_svd.py',
    'results/svd/',
    '--plot-spectra',
    '--condition-table'
], shell=True)
```

---

## Error Handling

### Common Exceptions

```python
try:
    processor = Z5ArrayProcessor(N=5, d=1.0)  # N=5 not supported
    results = processor.run_full_analysis()
except ValueError as e:
    print(f"Invalid parameter: {e}")

try:
    doas_est, info = estimate_doa_coarray_music(
        X=X, sensor_positions=[], wavelength=1.0, num_sources=2
    )
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

### Validation Patterns

```python
# Z4 processor with assertions
from geometry_processors.z4_processor_ import Z4ArrayProcessor

processor = Z4ArrayProcessor(N=7, d=1.0, do_asserts=True)
results = processor.run_full_analysis()

# Will raise AssertionError if:
#   - w(1) ≠ 0 or w(2) ≠ 0
#   - First segment length ≠ 3
```

---

## Integration Examples

### Batch Array Comparison

```python
from geometry_processors.ula_processors_ import ULArrayProcessor
from geometry_processors.nested_processor_ import NestedArrayProcessor
from geometry_processors.z5_processor_ import Z5ArrayProcessor
import pandas as pd

# Compare 3 arrays
arrays = {
    'ULA': ULArrayProcessor(M=7, d=1.0),
    'Nested': NestedArrayProcessor(N1=3, N2=4, d=1.0),
    'Z5': Z5ArrayProcessor(N=7, d=1.0)
}

results = []
for name, proc in arrays.items():
    spec = proc.run_full_analysis()
    results.append({
        'Array': name,
        'N': spec.num_sensors,
        'Mv': spec.num_virtual,
        'K_max': spec.max_detectable_sources,
        'Aperture': spec.coarray_aperture,
        'Holes': len(spec.holes)
    })

# Display comparison table
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

### Automated Parameter Sweep

```python
import pandas as pd
from geometry_processors.z5_processor_ import Z5ArrayProcessor

# Sweep spacing parameter
d_values = [0.5, 1.0, 1.5, 2.0]
results = []

for d in d_values:
    proc = Z5ArrayProcessor(N=7, d=d)
    spec = proc.run_full_analysis()
    results.append({
        'd': d,
        'Aperture': spec.coarray_aperture,
        'K_max': spec.max_detectable_sources,
        'W(1)': spec.weights_dict.get(1, 0)
    })

df = pd.DataFrame(results)
df.to_csv('results/summaries/spacing_sweep.csv', index=False)
```

---

**For full examples, see:** [tutorials/](tutorials/) directory

**For troubleshooting:** See [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)
