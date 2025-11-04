# Publication Workflow Guide

This document provides the complete workflow from benchmark execution to publication-ready figures and tables.

## Summary of Changes

### 1. CRB Interpretation Fix ✅
**Issue**: Summary statements incorrectly claimed performance "better than CRB" when ratios were actually above 1.0 (correct).

**Fix**: Added `crb_ratio_safestr()` helper function in `analysis_scripts/run_benchmarks.py`:
```python
def crb_ratio_safestr(rmse_deg: float, crb_deg: float) -> str:
    """Format RMSE/CRB ratio with correct interpretation (RMSE is always >= CRB)."""
    if crb_deg is None or crb_deg <= 0:
        return "CRB N/A"
    ratio = rmse_deg / crb_deg
    return f"{ratio:.2f}× above CRB"
```

**Impact**: All performance summaries now correctly report "X× above CRB" instead of misleading "better than CRB" language.

### 2. Root-MUSIC Marked as Experimental ⚠️
**Issue**: Root-MUSIC on virtual arrays produces unstable estimates (RMSE ~10-23° vs grid ~0.9°).

**Fix**: Added warning in `algorithms/coarray_music.py`:
```python
if use_root:
    print("[WARN] Coarray Root-MUSIC is experimental; grid search is recommended for stable results (see paper).")
```

**Documentation**: Added limitation note to `README.md`:
> Root-MUSIC on virtual arrays: Provided for research purposes only; use grid-based CoarrayMUSIC for reported results.

**Status**: Keep disabled by default (no `--use-root` flag in production runs).

### 3. Z6 Coarray Analysis 🔍
**Finding**: Z6 correctly reports Mv=3. This is NOT a bug - it's a fundamental geometry limitation.

**Root Cause**: Z6 gap pattern `[4,3,3,4,4,3,4,3]` (designed for w(1)=w(2)=0) creates fragmented coarray:
- Positions: `[0, 4, 7, 10, 14, 18, 21]`
- Positive lags: `[0, 3, 4, 6, 7, 8, 10, 11, 14, 17, 18, 21]`
- Longest contiguous: `[6, 7, 8]` (length 3)
- K_max = floor(3/2) = 1 source

**Scaling Analysis**: Mv stays at 3 even for N=15 sensors (doesn't grow!).

**Recommendation for Paper**:
- Exclude Z6 from CoarrayMUSIC plots
- Keep Z6 in SpatialMUSIC comparisons
- Add caveat: "Z6 optimized for w(1)=w(2)=0 yields fragmented coarray (Mv=3), unsuitable for virtual-array methods"

## Headline Benchmark Configuration

### Command
```powershell
.\mimo-geom-dev\Scripts\python.exe -m analysis_scripts.run_benchmarks `
  --arrays Z4,Z5,ULA `
  --algs SpatialMUSIC,CoarrayMUSIC `
  --N 7 --d 1.0 --lambda_factor 2.0 `
  --snr 0,5,10,15 `
  --snapshots 32,128,256,512 `
  --k 2 --delta 1,2,3 `
  --trials 100 `
  --out results/bench/headline.csv --save-crb
```

### Parameters Explained
- `--arrays Z4,Z5,ULA`: Three array types (Z6 excluded from coarray plots)
- `--algs SpatialMUSIC,CoarrayMUSIC`: Both algorithms for comparison
- `--N 7`: 7 sensors per array
- `--d 1.0 --lambda_factor 2.0`: d=1m, λ=2m → d=λ/2 (standard half-wavelength spacing)
- `--snr 0,5,10,15`: SNR sweep from 0-15dB
- `--snapshots 32,128,256,512`: Snapshot count sweep (low to high)
- `--k 2 --delta 1,2,3`: Two-source scenario with angular separation sweep
- `--trials 100`: 100 Monte Carlo trials per condition
- `--save-crb`: Generate CRB overlay file for comparison

### Expected Runtime
- Total configurations: 3 arrays × 2 algorithms × 4 SNR × 4 snapshots × 3 deltas × 100 trials = **28,800 runs**
- Estimated time: ~2-4 hours (depends on CPU)

### Outputs
- `results/bench/headline.csv`: Main benchmark results (21 columns, ~28,800 rows)
- `results/bench/crb_overlay.csv`: CRB values for each configuration

## Smoke Test (Fast Validation)

Before running the full benchmark, validate with reduced parameters:

```powershell
.\run_headline_smoke.ps1
```

This runs:
- 2 SNR levels (0, 10 dB)
- 2 snapshot counts (64, 256)
- 1 delta value (2°)
- 10 trials per condition
- Total: 3×2×2×2×1×10 = 240 runs (~2-5 minutes)

## Generate Publication Figures

After benchmark completes, generate figures:

```powershell
.\mimo-geom-dev\Scripts\python.exe scripts\plot_headline.py
```

### Outputs
1. **`results/figs/rmse_vs_M_SNR10_delta2.png`**: RMSE vs snapshots at SNR=10dB, Δθ=2°
2. **`results/figs/resolve_vs_SNR_M256_delta2.png`**: Resolution rate vs SNR at M=256, Δθ=2°
3. **`results/figs/heatmap_Z5_spatial.png`**: Z5 SpatialMUSIC RMSE heatmap (SNR×M) with CRB contours
4. **`results/figs/headline_table_SNR10_M256_delta2.csv`**: Summary table at headline point (SNR=10, M=256, Δθ=2)

### Figure Specifications
- Resolution: 200 DPI (publication quality)
- Format: PNG with tight bounding boxes
- Color scheme: Distinct markers/colors per array-algorithm pair
- CRB overlay: Cyan contours with labels on heatmap

## Summarize Results

View formatted summary with CRB ratios:

```powershell
.\summarize_headline.ps1
```

This displays:
- Full configuration sweep with RMSE, Resolve%, Mv, CRB, and "X× above CRB" ratios
- Highlighted headline point (SNR=10, M=256, Δθ=2)
- Correct interpretation notes

## Expected Benchmark Results

Based on production validation (`final_production_M256.csv`):

### Headline Point (SNR=10dB, M=256, Δθ=2°)

| Array | Algorithm | RMSE (°) | Resolve (%) | Mv | CRB (°) | RMSE/CRB |
|-------|-----------|----------|-------------|----|---------|---------:|
| Z4 | SpatialMUSIC | 0.279 | 82 | 7 | 0.048 | 5.81× |
| Z5 | SpatialMUSIC | **0.129** | **94** | 7 | 0.048 | **2.68×** |
| ULA | SpatialMUSIC | 0.947 | 30 | 7 | 0.048 | 19.7× |
| Z4 | CoarrayMUSIC | 0.905 | 70 | 12 | 0.024 | 37.7× |
| Z5 | CoarrayMUSIC | 0.904 | 62 | 10 | 0.024 | 37.7× |
| ULA | CoarrayMUSIC | 0.943 | 26 | 13 | 0.024 | 39.3× |

**Key Findings**:
- **Z5 SpatialMUSIC**: Best performer (0.129°, 94% resolve, 2.68× CRB)
- **Z4 SpatialMUSIC**: Robust to aliasing (0.279°, 82% resolve)
- **CoarrayMUSIC**: Stable but higher RMSE than SpatialMUSIC (~0.9° vs 0.1-0.3°)
- **ULA**: Poor performance due to limited aperture

## LaTeX Integration

### Table Import
```latex
\begin{table}[ht]
\centering
\caption{Headline Benchmark Results (SNR=10dB, M=256, $\Delta\theta=2^\circ$)}
\label{tab:headline}
\csvautotabular{results/figs/headline_table_SNR10_M256_delta2.csv}
\end{table}
```

### Figure Import
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{results/figs/rmse_vs_M_SNR10_delta2.png}
\caption{RMSE vs Snapshots at SNR=10dB, $\Delta\theta=2^\circ$}
\label{fig:rmse_vs_M}
\end{figure}
```

## Z6 Treatment in Paper

### Geometry Section
Present Z6 as a weight-constrained design case study:

> "Z6 employs a gap pattern [4,3,3,4,4,3,4,3] designed to eliminate w(1)=w(2)=0 (no 1- or 2-unit differences). While this achieves the weight constraints, it produces a highly fragmented difference coarray with longest contiguous segment Mv=3, limiting K_max to 1 detectable source."

### DOA Results Section
Exclude Z6 from CoarrayMUSIC plots. Include Z6 in SpatialMUSIC comparisons with note:

> "Z6 is omitted from CoarrayMUSIC benchmarks due to its fragmented coarray structure (Mv=3), which makes it unsuitable for virtual array DOA estimation methods."

### Design Trade-offs Discussion
Use Z6 as cautionary example:

> "The Z6 case illustrates an important design trade-off: optimizing for specific weight constraints (e.g., w(1)=w(2)=0) can inadvertently destroy the contiguous virtual aperture required for coarray-based processing, even as the physical array aperture grows with N."

## Files Modified

1. `analysis_scripts/run_benchmarks.py`: Added `crb_ratio_safestr()` helper
2. `algorithms/coarray_music.py`: Added Root-MUSIC experimental warning
3. `README.md`: Added limitations section
4. `scripts/plot_headline.py`: Publication plotting script (NEW)
5. `run_headline_smoke.ps1`: Fast validation script (NEW)
6. `summarize_headline.ps1`: Result summary script (NEW)
7. `PUBLICATION_WORKFLOW.md`: This document (NEW)

## Troubleshooting

### Benchmark Too Slow
Reduce parameters in smoke test:
```powershell
# Minimal test: 1 SNR, 1 M, 1 delta, 5 trials
.\mimo-geom-dev\Scripts\python.exe -m analysis_scripts.run_benchmarks `
  --arrays Z5 --algs SpatialMUSIC --N 7 --snr 10 --snapshots 128 --k 2 --delta 2 --trials 5 `
  --out results/bench/test.csv
```

### Plot Script Fails
Check inputs exist:
```powershell
Test-Path results/bench/headline.csv
Test-Path results/bench/crb_overlay.csv
```

If CRB file missing, plots will skip contours but still generate figures.

### Memory Issues
Process results in chunks by filtering CSV before plotting:
```python
# In plot_headline.py, add after loading:
df = df.query("SNR_dB.isin([0,5,10,15]) & snapshots.isin([32,128,256,512])")
```

## Next Steps for Publication

1. ✅ Run headline benchmark (3-4 hours)
2. ✅ Generate figures with `plot_headline.py`
3. ✅ Review summary with `summarize_headline.ps1`
4. ⏳ Write paper sections referencing figures/tables
5. ⏳ Add Z6 design trade-off discussion
6. ⏳ Submit supplementary materials (CSV files, scripts)

## Contact
For questions about this workflow, see `copilot-instructions.md` or project README.
