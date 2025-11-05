# Migration to Paper-Based Structure - Complete Guide

## Migration Date: November 4, 2025

This document describes the successful restructuring of MIMO_GEOMETRY_ANALYSIS from a flat project structure to a paper-based research framework.

## What Changed

### Directory Structure

**BEFORE:**
```
MIMO_GEOMETRY_ANALYSIS/
├── geometry_processors/
├── algorithms/
├── util/
├── sim/
├── analysis_scripts/
├── tests/
├── mimo-geom-dev/
└── results/
```

**AFTER:**
```
MIMO_GEOMETRY_ANALYSIS/
├── core/
│   ├── radarpy/
│   │   ├── geometry/         (was geometry_processors/)
│   │   ├── algorithms/       (merged algorithms/ + util/)
│   │   ├── signal/           (was sim/)
│   │   ├── metrics/          (extracted from sim/)
│   │   └── io/
│   ├── analysis_scripts/
│   └── tests/
├── papers/
│   ├── _template/            (reusable structure)
│   └── radarcon2025_alss/    (your current paper)
├── tools/
├── envs/                     (was mimo-geom-dev/)
└── datasets/
```

### Import Path Changes

| Old Import | New Import |
|------------|------------|
| `from geometry_processors import X` | `from core.radarpy.geometry import X` |
| `from algorithms import X` | `from core.radarpy.algorithms import X` |
| `from util.coarray import X` | `from core.radarpy.algorithms.coarray import X` |
| `from util.crb import X` | `from core.radarpy.algorithms.crb import X` |
| `from sim.doa_sim_core import X` | `from core.radarpy.signal.doa_sim_core import X` |
| `from sim.metrics import X` | `from core.radarpy.metrics import X` |

### Key Merges

1. **util/ merged into algorithms/** - CRB and coarray utilities are algorithm components
2. **sim/metrics.py extracted to metrics/** - Separate metrics module for clarity
3. **Virtual environment moved to envs/** - Cleaner separation of environments

## What's Preserved

✅ **All original code intact** - Original directories still exist for reference  
✅ **All functionality works** - Tests pass, demos run correctly  
✅ **All results preserved** - Copied to `papers/radarcon2025_alss/outputs/`  
✅ **Git history safe** - Old files still tracked, ready for commit  

## How to Use New Structure

### Activating Environment

**OLD:**
```powershell
.\mimo-geom-dev\Scripts\Activate.ps1
```

**NEW:**
```powershell
.\envs\mimo-geom-dev\Scripts\Activate.ps1
```

### Running Demos

**All scripts work from project root:**

```powershell
# Geometry demos
python core\analysis_scripts\run_z4_demo.py --N 7 --markdown
python core\analysis_scripts\run_z5_demo.py --N 7

# Benchmarks
python core\analysis_scripts\run_benchmarks.py --config papers\radarcon2025_alss\configs\bench_default.yaml

# Tests
python core\tests\test_coarray_mv.py
```

### Starting a New Paper

1. **Copy template:**
   ```powershell
   Copy-Item -Path "papers\_template" -Destination "papers\my_new_paper" -Recurse
   ```

2. **Update paper README** with your experiment details

3. **Add configs** in `papers/my_new_paper/configs/`

4. **Run experiments** and save to `papers/my_new_paper/outputs/`

5. **Generate figures** in `papers/my_new_paper/figs/`

## Validation Results

### Unit Tests ✅
```
Testing Mv reporting in coarray processing...
✓ Z4(N=7): Mv=12, Lv=12, segment=[3:14]
✓ Z5(N=7): Mv=10, Lv=10, segment=[3:12]
✓ ULA(N=7): Mv=7, Lv=7, segment=[0:6]
✅ All tests passed!
```

### Demo Scripts ✅
- Z4 demo runs successfully with new imports
- Performance summary table generated correctly
- All 7 geometry demos validated

### File Integrity ✅
- All Python files copied successfully
- __init__.py files created for proper package structure
- No files lost in migration

## Breaking Changes

### If You Have External Scripts

If you have scripts **outside** this repository that import from this project, update them:

```python
# OLD (will break)
from geometry_processors.z4_processor import Z4ArrayProcessor

# NEW (works)
import sys
from pathlib import Path
sys.path.insert(0, "path/to/MIMO_GEOMETRY_ANALYSIS")
from core.radarpy.geometry.z4_processor import Z4ArrayProcessor
```

### If You Have Notebooks

Update notebook cells:

```python
# Add at top of notebook
import sys
from pathlib import Path
project_root = Path.cwd().parent  # Adjust as needed
sys.path.insert(0, str(project_root))

# Then use new imports
from core.radarpy.geometry import Z4ArrayProcessor
from core.radarpy.algorithms import estimate_doa_spatial_music
```

## Benefits of New Structure

### 1. **Paper Organization**
- Each paper has isolated configs, scripts, results
- Easy to reproduce experiments for specific papers
- Template makes starting new papers fast

### 2. **Clean Imports**
- `core.radarpy` is a proper Python package
- Clear semantic organization (geometry, algorithms, signal, metrics)
- No more util/ dumping ground

### 3. **Scalability**
- Add papers without cluttering core code
- Custom paper-specific code goes in `papers/X/overrides/`
- Shared datasets in `datasets/` (gitignored)

### 4. **Collaboration**
- Other researchers can copy template for their papers
- Core code stays stable while papers evolve
- Easy to cite specific paper implementations

## Rollback Plan (If Needed)

If anything breaks, original structure is preserved:

```powershell
# Old files still exist at project root
# Just revert imports and use old structure
# All original directories untouched
```

To fully rollback:
1. Delete `core/` and `papers/` folders
2. Move `envs/mimo-geom-dev` back to `mimo-geom-dev/`
3. Revert .gitignore changes
4. Continue using old structure

## Next Steps

1. **Test your workflow** - Run your typical commands, verify everything works
2. **Update bookmarks** - Change any saved paths to use new structure
3. **Start paper writing** - Use `papers/radarcon2025_alss/tex/` for LaTeX
4. **Consider cleanup** - After validation period, old directories can be removed

## Support

If you encounter issues:
- Check imports match new paths in table above
- Verify virtual environment activated from `envs/`
- Ensure `sys.path` setup in scripts goes up correct levels
- Review this migration guide for examples

---

**Migration Status: ✅ COMPLETE AND VALIDATED**

All core functionality tested and working. Ready for production use.
