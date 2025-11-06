# MIMO Array Geometry Analysis Framework

## Overview
Analysis pipeline for MIMO radar array geometries through difference coarray computation. Supports ULA, Nested, and specialized Z-family arrays (Z1, Z3_1, Z3_2, Z4, Z5, Z6).

## Key Features
- **Difference Coarray Analysis**: Compute pairwise differences between sensor positions to derive virtual sensor arrays
- **Performance Metrics**: Coarray aperture, contiguous segment length, maximum detectable sources, weight distribution
- **DOA Estimation**: SpatialMUSIC (physical array) and CoarrayMUSIC (virtual array) implementations
- **Benchmark Infrastructure**: Comprehensive testing across array types, SNR, snapshot counts, source separations

## Installation
```powershell
# Activate virtual environment
.\mimo-geom-dev\Scripts\Activate.ps1

# Verify dependencies (numpy, pandas, matplotlib, scipy)
pip list
```

## Quick Start
```powershell
# Run geometry analysis
python analysis_scripts/run_z5_demo.py --N 7 --markdown

# Run DOA benchmark
python -m analysis_scripts.run_benchmarks --arrays Z4,Z5,ULA --N 7 --snr 0,10 --snapshots 64,256 --k 2 --delta 2 --trials 50 --out results/bench/test.csv
```

## Limitations
- **Root-MUSIC on virtual arrays**: Provided for research purposes only; use grid-based CoarrayMUSIC for reported results. Root-MUSIC implementation is experimental and may produce unstable estimates.
- **Z6 CoarrayMUSIC**: Z6 geometry is optimized for w(1)=w(2)=0 but produces a fragmented coarray (Mv=3), making it unsuitable for virtual array DOA methods.

## Citation
If you use this framework in your research, please cite our paper (details TBD).
