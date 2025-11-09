# MCM Effect Analysis - Quick Reference

## What This Analysis Shows

**Complete comparison of MCM ON vs OFF across all array geometries with ALL metrics**

## Run the Analysis

```bash
python analysis_scripts/compare_mcm_effects.py
```

## Results Location

- **CSV Summary**: `results/summaries/mcm_comparison_summary.csv`
- **Detailed Report**: `results/MCM_EFFECT_ANALYSIS.md`

## Key Findings (Quick View)

### MCM Degradation by Array

| Array | RMSE (No MCM) | RMSE (With MCM) | Degradation | Impact |
|-------|---------------|-----------------|-------------|--------|
| **Z3_2** | 10.195° | 16.200° | **1.59×** | ⚠️ Most affected |
| **Z1** | 6.143° | 8.457° | **1.38×** | ⚠ Moderate |
| **Z5** | 2.293° | 2.186° | **0.95×** | ✓ MCM helps! |
| **ULA** | 0.000° | 0.000° | inf | ✓ Perfect both |
| **TCA** | 0.000° | 0.000° | inf | ✓ Perfect both |
| **Z3_1** | 0.000° | 0.000° | inf | ✓ Perfect both |
| **Z4** | 0.000° | 0.000° | inf | ✓ Perfect both |

### All Metrics Tracked

For each array, the analysis computes:

1. **RMSE** (Root Mean Square Error) - Primary accuracy metric
2. **MAE** (Mean Absolute Error) - Average magnitude of errors
3. **Bias** - Systematic estimation offset
4. **Max Error** - Worst-case single estimation error
5. **Success Rate** - Percentage of successful estimates (all 100%)
6. **Standard Deviation** - Variance across trials
7. **Min/Max Range** - Best and worst trial results

### Surprising Results

**Z5 Array**: MCM actually **improves** performance!
- No MCM: 2.293° RMSE
- With MCM: 2.186° RMSE (5% better)
- Bias reduced from -1.780° to -0.203° (7.8× better!)

**Z3_2 Array**: Most vulnerable to coupling
- RMSE increases 59% (1.59× worse)
- MAE increases 48% 
- Max Error increases 64%

## Practical Recommendations

### Choose These Arrays for Robustness
✓ **Z4** - Large spacing, perfect even with MCM
✓ **TCA** - Coprime redundancy  
✓ **ULA** - Dense uniform structure

### Avoid in High-Coupling Environments
⚠️ **Z3_2** - 1.59× degradation
⚠ **Z1** - 1.38× degradation

### When to Enable MCM
- Realistic hardware simulation
- Sparse arrays (Z1, Z3_2)
- Known coupling coefficients
- Planning compensation algorithms

### When to Disable MCM
- Best-case benchmarking
- Robust arrays (ULA, TCA, Z4)
- High SNR (>15 dB)
- Algorithm development phase

## Test Configuration

```python
SNR = 10 dB          # Realistic noise
Snapshots = 200      # Limited observations  
Trials = 50          # Monte Carlo average
Sources = 3          # At [-30°, 0°, 30°]

MCM Model:
  Type: Exponential decay
  c1 = 0.3           # Coupling strength
  alpha = 0.5        # Decay rate
```

## Output Format

The script generates:

1. **Console Output**: Detailed per-array results with statistics
2. **CSV File**: Compact summary table for all arrays
3. **Key Findings**: Automatic analysis of most/least affected arrays

## Example Usage

### Test specific array with MCM
```bash
# Z3_2 (most affected)
python analysis_scripts/run_doa_demo.py --array z3_2 --N 6 --K 3 --enable-mcm --SNR 10

# Compare without MCM
python analysis_scripts/run_doa_demo.py --array z3_2 --N 6 --K 3 --SNR 10
```

### Run full comparison
```bash
# All arrays, all metrics, 50 trials each
python analysis_scripts/compare_mcm_effects.py
```

## Understanding the Results

### Degradation Factor

```
Degradation = RMSE_with_MCM / RMSE_no_MCM

< 1.0  = MCM improves performance (rare!)
1.0-1.2 = Minor degradation (acceptable)
1.2-1.5 = Moderate degradation (consider compensation)
> 1.5  = Significant degradation (avoid or compensate)
inf    = Both perfect (0.000° / 0.000°)
```

### Bias Interpretation

```
Positive bias = Overestimation (angles too high)
Negative bias = Underestimation (angles too low)
Low bias = Accurate centering
```

### Standard Deviation

High std dev indicates **occasional complete failures**:
- Z3_2: ±21-27° → Some trials hit 70° error
- Z1: ±17-21° → High failure rate at SNR=10dB

## Related Files

- **MCM Implementation**: `doa_estimation/music.py` (lines 165-260)
- **MCM Test**: `analysis_scripts/test_mcm_doa.py`
- **CLI Demo**: `analysis_scripts/run_doa_demo.py` (--enable-mcm flag)
- **Documentation**: `doa_estimation/README.md` (MCM section)

## Citation

```
MIMO Geometry Analysis Framework
DOA Estimation Module with MCM Support
Version: 1.0 (2025-11-07)
```

---

**For detailed analysis and explanations, see**: `results/MCM_EFFECT_ANALYSIS.md`
