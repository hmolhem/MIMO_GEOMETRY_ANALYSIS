# RadarCon 2025: Aliasing-Limited Sparse Sensing (ALSS) Arrays

This paper investigates Z4, Z5, and Z6 array geometries for DOA estimation with focus on aliasing suppression and sparse sensing.

## Experiment Overview

**Benchmark Configuration:**
- Arrays: Z4, Z5, ULA (N=7 sensors)
- Algorithms: Spatial MUSIC, Coarray MUSIC
- SNR: [0, 5, 10, 15] dB
- Snapshots: [64, 128, 256, 512]
- Source separation: [2°, 4°, 8°]
- Trials per condition: 100

**Total experiments:** 28,800 trials (9,600 CSV rows)

## Key Findings

- **Z5 SpatialMUSIC**: Best performer (RMSE=0.185°, 87.9% resolve, 2.68× CRB)
- **Z4 SpatialMUSIC**: Robust to aliasing (RMSE=0.359°, 75.8% resolve)
- **Mv Bug Fixed**: Coarray processing now correctly reports virtual array size

## Reproduction

1. **Activate environment:**
   ```powershell
   .\envs\mimo-geom-dev\Scripts\Activate.ps1
   ```

2. **Run benchmark:**
   ```powershell
   python core\analysis_scripts\run_benchmarks.py --config papers\radarcon2025_alss\configs\bench_default.yaml
   ```

3. **Generate figures:**
   ```powershell
   python papers\radarcon2025_alss\scripts\plot_headline.py
   ```

## Files

- `configs/bench_default.yaml` - Benchmark parameters
- `scripts/plot_headline.py` - Publication figure generator
- `outputs/bench/headline.csv` - Raw results (9,600 rows)
- `figs/*.png` - Publication figures (200 DPI)
- `tex/` - LaTeX source (TBD)

## Status

✅ Experiments complete  
✅ Figures generated  
✅ CRB interpretation fixed  
✅ Mv reporting bug fixed  
⏳ Paper writing in progress
