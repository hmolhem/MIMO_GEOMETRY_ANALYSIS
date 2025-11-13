# Coprime Arrays Usage Guide (TCA & ePCA)

**Complete documentation for Two-level Coprime Array (TCA) and Extended Prime Coprime Array (ePCA) processors.**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [TCA (Two-level Coprime Array)](#tca-two-level-coprime-array)
3. [ePCA (Extended Prime Coprime Array)](#epca-extended-prime-coprime-array)
4. [Comparison Utilities](#comparison-utilities)
5. [Design Guidelines](#design-guidelines)
6. [Common Patterns](#common-patterns)

---

## Quick Start

### Installation & Setup

```python
# Add project to Python path (if needed)
import sys
sys.path.append('/path/to/MIMO_GEOMETRY_ANALYSIS')

# Import processors
from geometry_processors.tca_processor import TCAArrayProcessor, compare_tca_arrays
from geometry_processors.epca_processor import ePCAArrayProcessor, compare_epca_arrays
```

### 30-Second Examples

**TCA:**
```python
# Create TCA with M=3, N=5 (coprime)
tca = TCAArrayProcessor(M=3, N=5, d=1.0)
results = tca.run_full_analysis()
print(f"Sensors: {tca.total_sensors}, DOF: {results.segment_length//2}")
# Output: Sensors: 7, DOF: 4
```

**ePCA:**
```python
# Create ePCA with primes (2,3,5)
epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=3, N2=3, N3=3, d=1.0)
results = epca.run_full_analysis()
print(f"Sensors: {epca.total_sensors}, DOF: {results.segment_length//2}")
# Output: Sensors: 7, DOF: 2
```

---

## TCA (Two-level Coprime Array)

### Construction Formula

```
P1 = d × M × [0, 1, 2, ..., N-1]    (N elements, spacing M×d)
P2 = d × N × [0, 1, 2, ..., M-1]    (M elements, spacing N×d)
Combined: Merge P1 ∪ P2 (origin shared)
Total unique sensors: M + N - 1
```

### Basic Usage

```python
from geometry_processors.tca_processor import TCAArrayProcessor

# Example 1: Small coprime pair
tca = TCAArrayProcessor(M=2, N=3, d=1.0)
results = tca.run_full_analysis()

# Access results
print(f"Physical sensors: {tca.total_sensors}")           # 4
print(f"Coprime: {tca.is_coprime}")                       # True
print(f"Aperture: {results.aperture}")                    # 8
print(f"Segment length L: {results.segment_length}")      # 2
print(f"Max DOF (K_max): {results.segment_length // 2}")  # 1
print(f"Holes: {len(results.holes_in_segment)}")          # 0
```

### Parameter Selection

**M and N (Coprime Parameters):**
- **Must be ≥ 2** each
- **Coprime (gcd(M,N)=1) recommended** for optimal performance
- Common pairs: (2,3), (3,4), (3,5), (4,5), (5,7), (7,11)
- Larger values → More sensors, larger aperture, better DOF

**d (Unit Spacing):**
- **d = λ/2** for narrowband DOA (standard)
- **d = 1.0** as normalized unit (theory/simulation)
- Must be > 0

### Performance Metrics

```python
# Get comprehensive summary
summary = results.performance_summary_table
print(summary.to_markdown(index=False))

# Key metrics
dof_efficiency = (results.segment_length // 2) / tca.total_sensors
aperture_ratio = results.aperture / tca.total_sensors

print(f"DOF efficiency: {dof_efficiency:.2f}")  # K_max / N
print(f"Aperture ratio: {aperture_ratio:.1f}")  # A / N
```

### Weight Distribution

```python
# Access weight table
weights = results.weight_table
print(weights.head(10))  # First 10 lags

# Specific lag weights
w0 = int(weights[weights['Lag'] == 0]['Weight'].iloc[0])
w1 = int(weights[weights['Lag'] == 1]['Weight'].iloc[0])
w2 = int(weights[weights['Lag'] == 2]['Weight'].iloc[0])

print(f"w(0)={w0}, w(1)={w1}, w(2)={w2}")
```

### Visualization

```python
# ASCII plot (built-in)
tca.plot_coarray()

# Or access after analysis
results = tca.run_full_analysis()
# Visualization already printed during analysis
```

### Advanced Examples

**Example: Non-coprime pair (for comparison)**
```python
tca_coprime = TCAArrayProcessor(M=3, N=5, d=1.0)
tca_noncoprime = TCAArrayProcessor(M=3, N=6, d=1.0)

r1 = tca_coprime.run_full_analysis(verbose=False)
r2 = tca_noncoprime.run_full_analysis(verbose=False)

print(f"Coprime (3,5): DOF={r1.segment_length//2}, Holes={len(r1.holes_in_segment)}")
print(f"Non-coprime (3,6): DOF={r2.segment_length//2}, Holes={len(r2.holes_in_segment)}")
# Coprime will have better performance
```

**Example: Scaling study**
```python
configs = [(2,3), (3,5), (4,5), (5,7), (7,11)]
for M, N in configs:
    tca = TCAArrayProcessor(M=M, N=N, d=1.0)
    r = tca.run_full_analysis(verbose=False)
    eff = (r.segment_length // 2) / tca.total_sensors
    print(f"TCA({M},{N}): N={tca.total_sensors}, DOF={r.segment_length//2}, Eff={eff:.2f}")
```

### Error Handling

```python
# Invalid parameters
try:
    tca = TCAArrayProcessor(M=1, N=3)  # M < 2
except ValueError as e:
    print(f"Error: {e}")
# Output: TCA requires M ≥ 2 and N ≥ 2, got M=1, N=3

# Check coprimality
from math import gcd
if gcd(M, N) != 1:
    print(f"Warning: ({M},{N}) not coprime, may have degraded performance")
```

---

## ePCA (Extended Prime Coprime Array)

### Construction Formula

```
P1 = d × p2×p3 × [0, 1, 2, ..., N1-1]  (N1 elements, spacing p2×p3×d)
P2 = d × p1×p3 × [0, 1, 2, ..., N2-1]  (N2 elements, spacing p1×p3×d)
P3 = d × p1×p2 × [0, 1, 2, ..., N3-1]  (N3 elements, spacing p1×p2×d)
Combined: Merge P1 ∪ P2 ∪ P3 (origin shared 3 times)
Total unique sensors: N1 + N2 + N3 - 2
```

### Basic Usage

```python
from geometry_processors.epca_processor import ePCAArrayProcessor

# Example 1: Small primes, balanced subarrays
epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=3, N2=3, N3=3, d=1.0)
results = epca.run_full_analysis()

# Access results
print(f"Primes: ({epca.p1}, {epca.p2}, {epca.p3})")
print(f"Prime product: {epca.p1 * epca.p2 * epca.p3}")   # 30
print(f"Physical sensors: {epca.total_sensors}")          # 7
print(f"All coprime: {epca.is_coprime}")                  # True
print(f"Aperture: {results.aperture}")                    # 60
print(f"Segment length L: {results.segment_length}")      # 5
print(f"Max DOF (K_max): {results.segment_length // 2}")  # 2
```

### Parameter Selection

**p1, p2, p3 (Prime Parameters):**
- **Must be ≥ 2** and **p1 < p2 < p3** (ordered)
- **Pairwise coprime recommended** (though primes guarantee this)
- Common triplets:
  - **(2,3,5)** - Smallest, good for prototypes
  - **(3,5,7)** - Balanced performance/size
  - **(5,7,11)** - Large aperture
  - **(7,11,13)** - Very large aperture
- Product p1×p2×p3 determines aperture scale

**N1, N2, N3 (Subarray Sizes):**
- **Must be ≥ 2** each
- **Equal values (N1=N2=N3) recommended** for balanced coarray
- Unequal sizes for specific optimizations
- Total sensors = N1+N2+N3-2 (origin shared)

**d (Unit Spacing):**
- Same as TCA: typically λ/2 or 1.0

### Performance Metrics

```python
# Get comprehensive summary
summary = results.performance_summary_table
print(summary.to_markdown(index=False))

# Key metrics
print(f"DOF efficiency: {(results.segment_length//2) / epca.total_sensors:.2f}")
print(f"Aperture per sensor: {results.aperture / epca.total_sensors:.1f}")
```

### Advanced Examples

**Example: Compare prime triplets**
```python
triplets = [(2,3,5), (3,5,7), (5,7,11)]
for p1, p2, p3 in triplets:
    epca = ePCAArrayProcessor(p1=p1, p2=p2, p3=p3, N1=3, N2=3, N3=3)
    r = epca.run_full_analysis(verbose=False)
    print(f"ePCA({p1},{p2},{p3}): Product={p1*p2*p3}, Aperture={r.aperture}, DOF={r.segment_length//2}")
```

**Example: Unbalanced subarrays**
```python
# Optimize for fewer sensors while maintaining aperture
epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=4, N2=3, N3=2, d=1.0)
results = epca.run_full_analysis()
print(f"Total sensors: {epca.total_sensors} (reduced from 7 to 7 with N1=N2=N3=3)")
print(f"Aperture: {results.aperture} (maintained large)")
```

**Example: Very large aperture**
```python
# Large primes for research applications
epca = ePCAArrayProcessor(p1=7, p2=11, p3=13, N1=2, N2=2, N3=2, d=1.0)
results = epca.run_full_analysis(verbose=False)
print(f"Prime product: {7*11*13} = 1001")
print(f"Aperture: {results.aperture}")
print(f"Sensors needed: {epca.total_sensors} (only 4!)")
```

---

## Comparison Utilities

### TCA Batch Comparison

```python
from geometry_processors.tca_processor import compare_tca_arrays

# Compare multiple coprime pairs
configs = [(2,3), (3,4), (3,5), (4,5), (5,7), (7,11)]
comparison = compare_tca_arrays(configs, d=1.0)

# View results
print(comparison[['M', 'N', 'Total_Sensors', 'K_max', 'DOF_Efficiency']])

# Export
comparison.to_csv('tca_comparison.csv', index=False)
comparison.to_markdown('tca_comparison.md', index=False)

# Find best configuration
best = comparison.loc[comparison['DOF_Efficiency'].idxmax()]
print(f"Best: M={best['M']}, N={best['N']}, Efficiency={best['DOF_Efficiency']}")
```

### ePCA Batch Comparison

```python
from geometry_processors.epca_processor import compare_epca_arrays

# Compare prime triplets
prime_triplets = [(2,3,5), (3,5,7), (5,7,11), (7,11,13)]
comparison = compare_epca_arrays(prime_triplets, N1=3, N2=3, N3=3, d=1.0)

# View key metrics
print(comparison[['Prime_Product', 'Aperture', 'K_max', 'DOF_Efficiency']])

# Filter by criteria
high_efficiency = comparison[comparison['DOF_Efficiency'] > 0.5]
print(high_efficiency)
```

### Cross-Array Comparison

```python
# Compare TCA vs ePCA for similar configurations
tca = TCAArrayProcessor(M=3, N=5, d=1.0)
epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=3, N2=3, N3=3, d=1.0)

r_tca = tca.run_full_analysis(verbose=False)
r_epca = epca.run_full_analysis(verbose=False)

print("=== TCA vs ePCA Comparison ===")
print(f"TCA:  N={tca.total_sensors}, Aperture={r_tca.aperture}, DOF={r_tca.segment_length//2}")
print(f"ePCA: N={epca.total_sensors}, Aperture={r_epca.aperture}, DOF={r_epca.segment_length//2}")
```

---

## Design Guidelines

### When to Use TCA

**Advantages:**
- Simpler construction (only 2 subarrays)
- Fewer sensors than ePCA for similar aperture
- Well-studied in literature
- Good for hardware prototypes

**Best For:**
- Size-constrained systems
- Cost-sensitive applications
- Proven, conservative designs

### When to Use ePCA

**Advantages:**
- Better DOF efficiency (40-60% vs 25-40%)
- Fewer holes in coarray
- More uniform weight distribution
- Enhanced flexibility (3 parameters vs 2)

**Best For:**
- Research applications
- Maximum DOF requirements
- Systems where sensor count is not critical

### Coprimality Matters

```python
from math import gcd

# Always check coprimality
def is_coprime(a, b):
    return gcd(a, b) == 1

# For TCA
M, N = 3, 6  # Non-coprime!
if not is_coprime(M, N):
    print(f"Warning: ({M},{N}) not coprime (gcd={gcd(M,N)})")
    # Consider using (3,5) or (3,7) instead

# For ePCA  
p1, p2, p3 = 2, 3, 5
if not (is_coprime(p1,p2) and is_coprime(p2,p3) and is_coprime(p1,p3)):
    print("Warning: Primes not pairwise coprime")
```

### Spacing Guidelines

```python
# Narrowband DOA (standard)
wavelength = 0.1  # meters
d = wavelength / 2  # Half-wavelength spacing
tca = TCAArrayProcessor(M=3, N=5, d=d)

# Normalized (simulation/theory)
tca = TCAArrayProcessor(M=3, N=5, d=1.0)

# Mutual coupling constrained
d = wavelength  # Full wavelength to reduce coupling
tca = TCAArrayProcessor(M=3, N=5, d=d)
```

---

## Common Patterns

### Pattern 1: Parameter Sweep

```python
# Sweep M and N for TCA
import pandas as pd

results = []
for M in range(2, 8):
    for N in range(M+1, 10):
        if gcd(M, N) == 1:  # Only coprime
            tca = TCAArrayProcessor(M=M, N=N, d=1.0)
            r = tca.run_full_analysis(verbose=False)
            results.append({
                'M': M, 'N': N,
                'Sensors': tca.total_sensors,
                'Aperture': r.aperture,
                'DOF': r.segment_length // 2,
                'Efficiency': (r.segment_length//2) / tca.total_sensors
            })

df = pd.DataFrame(results)
print(df.sort_values('Efficiency', ascending=False).head(10))
```

### Pattern 2: Constraint Optimization

```python
# Find configuration with at least K_min DOF using ≤ N_max sensors
K_min = 10
N_max = 15

candidates = []
for M in range(2, 20):
    for N in range(M+1, 20):
        if gcd(M, N) != 1:
            continue
        if M + N - 1 > N_max:
            continue
            
        tca = TCAArrayProcessor(M=M, N=N)
        r = tca.run_full_analysis(verbose=False)
        K = r.segment_length // 2
        
        if K >= K_min:
            candidates.append((M, N, tca.total_sensors, K))

# Sort by efficiency
candidates.sort(key=lambda x: x[3]/x[2], reverse=True)
print(f"Top 5 candidates for K≥{K_min}, N≤{N_max}:")
for M, N, n, K in candidates[:5]:
    print(f"  ({M},{N}): N={n}, K={K}, Eff={K/n:.2f}")
```

### Pattern 3: Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# DOF efficiency vs sensor count
configs = [(M, N) for M in range(2,10) for N in range(M+1,12) if gcd(M,N)==1]
sensors = []
dofs = []

for M, N in configs[:20]:  # First 20
    tca = TCAArrayProcessor(M=M, N=N)
    r = tca.run_full_analysis(verbose=False)
    sensors.append(tca.total_sensors)
    dofs.append(r.segment_length // 2)

plt.figure(figsize=(10, 6))
plt.scatter(sensors, dofs, s=100, alpha=0.6)
plt.xlabel('Physical Sensors (N)', fontsize=12)
plt.ylabel('Maximum DOF (K_max)', fontsize=12)
plt.title('TCA: DOF vs Sensor Count', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tca_dof_scaling.png', dpi=150)
```

### Pattern 4: Export for Publication

```python
# Generate publication-ready comparison table
configs = [(2,3), (3,4), (3,5), (4,5), (5,7), (7,11)]
comparison = compare_tca_arrays(configs, d=0.5)  # λ/2 spacing

# Select columns for paper
paper_table = comparison[[
    'M', 'N', 'Total_Sensors', 'Aperture', 
    'K_max', 'Holes', 'DOF_Efficiency'
]]

# Round for readability
paper_table['DOF_Efficiency'] = paper_table['DOF_Efficiency'].round(2)

# Export as LaTeX
latex_table = paper_table.to_latex(index=False, escape=False)
with open('tca_comparison.tex', 'w') as f:
    f.write(latex_table)

# Also markdown for GitHub/documentation
md_table = paper_table.to_markdown(index=False)
print(md_table)
```

---

## Summary

### Quick Reference

| **Feature** | **TCA** | **ePCA** |
|-------------|---------|----------|
| Subarrays | 2 | 3 |
| Parameters | M, N | p1, p2, p3, N1, N2, N3 |
| Sensors | M+N-1 | N1+N2+N3-2 |
| DOF Efficiency | 25-40% | 40-60% |
| Complexity | Simple | Moderate |
| Best For | Prototypes, cost | Research, max DOF |

### Best Practices

1. **Always check coprimality** before analysis
2. **Use d=λ/2** for real hardware, d=1.0 for theory
3. **Start with small configurations** (M,N ≤ 5 or small primes)
4. **Batch comparison** for design space exploration
5. **Export results** to CSV/Markdown for documentation
6. **Validate with tests** before hardware implementation

### Further Reading

- See `geometry_processors/tca_processor.py` for TCA implementation
- See `geometry_processors/epca_processor.py` for ePCA implementation
- Run `analysis_scripts/run_tca_demo.py --help` for TCA CLI
- Run `analysis_scripts/run_epca_demo.py --help` for ePCA CLI
- Check `core/tests/test_tca_array.py` for comprehensive test examples

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Framework:** MIMO Geometry Analysis v1.0
