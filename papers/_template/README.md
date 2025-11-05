# Paper Template

This template provides a reusable structure for implementing radar papers.

## Directory Structure

```
_template/
├── overrides/          # Custom array geometries or algorithm variants
├── hooks/              # Pre/post-processing hooks for experiments
├── configs/            # Configuration files (YAML/JSON)
├── scripts/            # Analysis and plotting scripts
├── outputs/            # Results, CSV files, logs
├── figs/               # Publication-ready figures
└── tex/                # LaTeX source files, tables, BibTeX
```

## Usage

1. **Copy this template** to create a new paper project:
   ```bash
   cp -r papers/_template papers/your_paper_name
   ```

2. **Customize configs/** with your experiment parameters

3. **Add custom code** to overrides/ if needed (new arrays, algorithms)

4. **Run experiments** using scripts/

5. **Generate figures** and save to figs/

6. **Write paper** in tex/

## Integration with Core

All template code should import from `core.radarpy`:

```python
from core.radarpy.geometry import Z4ArrayProcessor
from core.radarpy.algorithms import estimate_doa_spatial_music
from core.radarpy.signal import simulate_snapshots
```

## Configuration Best Practices

- Store all experiment parameters in `configs/*.yaml`
- Use consistent naming: `{experiment}_{variant}.yaml`
- Document parameter choices in comments
- Version control all config files

## Output Organization

- `outputs/bench/` - Raw benchmark CSV files
- `outputs/summaries/` - Aggregated results
- `outputs/logs/` - Execution logs
- `figs/` - Only publication-ready plots (PNG/PDF, 300+ DPI)
