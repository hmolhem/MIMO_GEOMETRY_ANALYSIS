# MIMO Array Geometry Analysis Framework

A comprehensive research framework for analyzing MIMO radar array geometries through difference coarray analysis, DOA estimation, and performance benchmarking.

**Version:** 2.0.0 (Paper-Based Structure)  
**Last Updated:** November 4, 2025

## Quick Start

```powershell
# 1. Activate environment
.\envs\mimo-geom-dev\Scripts\Activate.ps1

# 2. Run a geometry demo
python core\analysis_scripts\run_z4_demo.py --N 7 --markdown

# 3. Run unit tests
python core\tests\test_coarray_mv.py

# 4. Run benchmarks (if configured)
python core\analysis_scripts\run_benchmarks.py
```

## Project Structure

```
MIMO_GEOMETRY_ANALYSIS/
├── core/                      # Core research framework
│   ├── radarpy/              # Main Python package
│   │   ├── geometry/         # Array processors (ULA, Nested, Z1-Z6)
│   │   ├── algorithms/       # DOA algorithms (MUSIC variants, CRB, coarray)
│   │   ├── signal/           # Signal simulation and array manifold
│   │   ├── metrics/          # Performance metrics (RMSE, resolution)
│   │   └── io/               # Input/output utilities
│   ├── analysis_scripts/     # CLI demos and benchmarks
│   └── tests/                # Unit tests
├── papers/                    # Paper-specific research
│   ├── _template/            # Reusable paper template
│   └── radarcon2025_alss/    # Example: RadarCon 2025 paper
│       ├── configs/          # Experiment configurations
│       ├── scripts/          # Paper-specific scripts
│       ├── outputs/          # Results (CSV, logs)
│       ├── figs/             # Publication figures
│       └── tex/              # LaTeX source
├── tools/                     # Helper scripts
├── envs/                      # Virtual environments (gitignored)
├── datasets/                  # Large datasets (gitignored)
└── README.md                  # This file
```

## Core Components

### 1. Array Geometries (`core/radarpy/geometry/`)

8 MIMO array implementations with standardized analysis pipeline:

- **ULA** - Uniform Linear Array (baseline)
- **Nested Array** - Contiguous coarray design
- **Z1** - 2-Sparse ULA + 1 Sensor (w(1)=0)
- **Z3(1)** - 4-Sparse ULA + 3 Sensors (w(1)=0, w(2)=2)
- **Z3(2)** - Variant with w(2)=1
- **Z4** - w(1)=w(2)=0 constraint
- **Z5** - Advanced w(1)=w(2)=0 (best performer in benchmarks)
- **Z6** - Ultimate weight constraints

Each processor follows a 7-step analysis:
1. Physical array specification
2. Difference coarray computation
3. Coarray analysis (unique elements, holes, segments)
4. Weight distribution
5. Contiguous segment analysis
6. Holes analysis
7. Performance summary

### 2. DOA Algorithms (`core/radarpy/algorithms/`)

**Spatial MUSIC** - Standard MUSIC on physical array
- Near-CRB performance at high SNR
- Robust to aliasing (especially Z4/Z5)
- Fast, no coarray processing overhead

**Coarray MUSIC** - MUSIC on virtual ULA covariance
- Leverages difference coarray structure
- Includes FBA, unbiased lag averaging, diagonal loading
- Grid search (stable) or Root-MUSIC (experimental)

**CRB Computation** - Cramér-Rao Bound for DOA estimation
- Single-source/pair bounds
- Used for performance comparison

### 3. Simulation & Metrics (`core/radarpy/signal/`, `core/radarpy/metrics/`)

- Narrowband snapshot generation
- Array manifold (steering vectors)
- RMSE computation
- Resolution indicators
- Success rate metrics

## Usage Examples

### Geometry Analysis

```python
from core.radarpy.geometry import Z5ArrayProcessor

# Create and analyze Z5 array
processor = Z5ArrayProcessor(N=7, d=0.5)
results = processor.run_full_analysis()

# Access results
print(results.performance_summary_table.to_markdown())
print(f"Max detectable sources: {results.K_max}")
print(f"Contiguous segment: {results.largest_onesided_segment}")
```

### DOA Estimation

```python
from core.radarpy.algorithms import estimate_doa_spatial_music
from core.radarpy.signal import simulate_snapshots

# Simulate data
positions = [0, 5, 8, 11, 14, 17, 21]  # Z4 N=7
wavelength = 0.1
doas_true = [-10.0, 5.0]  # Two sources
X = simulate_snapshots(positions, d=0.5, wavelength, doas_true, 
                       snapshots=256, snr_db=10)

# Estimate DOA
doas_est = estimate_doa_spatial_music(X, positions, d=0.5, 
                                       wavelength, K=2)
print(f"True DOA: {doas_true}")
print(f"Estimated: {doas_est}")
```

### Benchmarking

```python
# Via CLI (recommended)
python core/analysis_scripts/run_benchmarks.py \
    --array-type Z5 \
    --algorithm spatial \
    --N 7 \
    --SNR 0 5 10 15 \
    --snapshots 64 128 256 512 \
    --trials 100 \
    --output papers/my_paper/outputs/bench/results.csv
```

## Paper Workflow

### Starting a New Paper

1. **Copy template:**
   ```powershell
   Copy-Item -Recurse papers\_template papers\my_paper_name
   ```

2. **Configure experiment:**
   Edit `papers/my_paper_name/configs/experiment.yaml`

3. **Run experiments:**
   Save outputs to `papers/my_paper_name/outputs/`

4. **Generate figures:**
   Use scripts in `papers/my_paper_name/scripts/`
   Save to `papers/my_paper_name/figs/`

5. **Write paper:**
   LaTeX source in `papers/my_paper_name/tex/`

### Example: RadarCon 2025 Paper

See `papers/radarcon2025_alss/README.md` for complete example with:
- 28,800 trial benchmark
- Publication-ready figures
- Configuration files
- Results validation

## Installation

### Requirements

- Python 3.13.0 (via pyenv)
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0

### Setup

```powershell
# Create virtual environment (if not exists)
python -m venv envs\mimo-geom-dev

# Activate
.\envs\mimo-geom-dev\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Key Features

✅ **8 Array Geometries** - ULA, Nested, Z1-Z6 with standardized analysis  
✅ **2 MUSIC Variants** - Spatial (stable) and Coarray (experimental Root-MUSIC)  
✅ **Comprehensive Benchmarking** - SNR, snapshots, separation sweeps  
✅ **Paper Organization** - Isolated experiments per paper  
✅ **Production-Ready** - Unit tests, validation, CRB comparison  
✅ **Publication Tools** - Figure generation, LaTeX tables, result export

## Limitations

### Root-MUSIC (Experimental)
- High variance (~10-23° RMSE vs 0.9° grid search)
- Numerical instability with polynomial roots
- **Recommendation:** Use grid search for production

### Z6 Array
- Fragmented coarray (Mv=3)
- Not suitable for Coarray MUSIC
- Documented limitation, not a bug

See `papers/radarcon2025_alss/README.md` for detailed findings.

## Migration Notes

**Migrated from flat structure (v1.0) on November 4, 2025**

Key changes:
- `geometry_processors/` → `core/radarpy/geometry/`
- `algorithms/` + `util/` → `core/radarpy/algorithms/`
- `sim/` → `core/radarpy/signal/`
- `mimo-geom-dev/` → `envs/mimo-geom-dev/`

See `MIGRATION_GUIDE.md` for complete details and rollback instructions.

## Testing

```powershell
# Unit tests
python core\tests\test_coarray_mv.py

# Geometry demos (all arrays)
python core\analysis_scripts\run_z4_demo.py --N 7
python core\analysis_scripts\run_z5_demo.py --N 7
python core\analysis_scripts\run_ula_demo.py --M 7

# Comprehensive method tests
python core\analysis_scripts\methods_demo.py
```

Expected output: All tests pass ✅

## Contributing

When adding new features:
1. Add array geometries to `core/radarpy/geometry/`
2. Add algorithms to `core/radarpy/algorithms/`
3. Add tests to `core/tests/`
4. Update this README

For paper-specific code:
1. Use `papers/your_paper/overrides/` for custom arrays/algorithms
2. Keep core framework generic
3. Document paper-specific modifications

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{radar2025alss,
  title={Aliasing-Limited Sparse Sensing Arrays for DOA Estimation},
  author={Your Name},
  booktitle={IEEE RadarCon},
  year={2025}
}
```

(Update with your actual paper details)

## License

[Add your license here]

## Support

- **Issues:** Report bugs or request features via issue tracker
- **Documentation:** See `papers/_template/README.md` for template usage
- **Migration Help:** See `MIGRATION_GUIDE.md` for structure changes

---

**Framework Status:** ✅ Production-ready  
**Latest Validation:** November 4, 2025  
**Test Coverage:** Core algorithms and geometries
