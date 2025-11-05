# Quick Reference: New Structure

## ⚡ Quick Commands

### Environment Setup
```powershell
# Activate (NEW PATH!)
.\envs\mimo-geom-dev\Scripts\Activate.ps1
```

### Run Tests
```powershell
# Unit tests
python core\tests\test_coarray_mv.py

# Demo scripts
python core\analysis_scripts\run_z4_demo.py --N 7 --markdown
python core\analysis_scripts\run_z5_demo.py --N 7
```

### Run Benchmarks
```powershell
python core\analysis_scripts\run_benchmarks.py \
    --config papers\radarcon2025_alss\configs\bench_default.yaml
```

## 📁 Key Paths

| What | Old Path | New Path |
|------|----------|----------|
| Virtual Env | `.\mimo-geom-dev\` | `.\envs\mimo-geom-dev\` |
| Geometry | `geometry_processors\` | `core\radarpy\geometry\` |
| Algorithms | `algorithms\`, `util\` | `core\radarpy\algorithms\` |
| Signal | `sim\` | `core\radarpy\signal\` |
| Scripts | `analysis_scripts\` | `core\analysis_scripts\` |
| Tests | `tests\` | `core\tests\` |
| Results | `results\` | `papers\radarcon2025_alss\outputs\` |

## 🔧 Import Changes

```python
# OLD
from geometry_processors.z4_processor import Z4ArrayProcessor
from util.coarray import build_virtual_ula_covariance
from sim.doa_sim_core import run_music

# NEW
from core.radarpy.geometry.z4_processor import Z4ArrayProcessor
from core.radarpy.algorithms.coarray import build_virtual_ula_covariance
from core.radarpy.signal.doa_sim_core import run_music
```

## 📄 Paper Workflow

1. **Copy template:** `Copy-Item -Recurse papers\_template papers\my_paper`
2. **Edit config:** `papers\my_paper\configs\experiment.yaml`
3. **Run experiments:** Save to `papers\my_paper\outputs\`
4. **Generate figures:** Save to `papers\my_paper\figs\`
5. **Write paper:** LaTeX in `papers\my_paper\tex\`

## ✅ Validation Status

- ✅ Unit tests pass (Z4, Z5, ULA Mv reporting correct)
- ✅ Z4 demo runs successfully
- ✅ Import paths all updated
- ✅ Virtual environment works from `envs/`
- ✅ All original files preserved

## 📚 Documentation

- **RESTRUCTURE_COMPLETE.md** - This migration summary
- **MIGRATION_GUIDE.md** - Detailed reference & rollback
- **README_NEW.md** - Complete framework docs
- **papers/_template/README.md** - Template usage

## 🆘 Quick Help

**Tests fail?**
- Check: `.\envs\mimo-geom-dev\Scripts\Activate.ps1`
- Verify: `python --version` shows 3.13.0

**Imports fail?**
- Use: `from core.radarpy.X import Y`
- Add: `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`

**Need rollback?**
- See: MIGRATION_GUIDE.md "Rollback Plan"
- Original files untouched in root directories

## 🎯 Next Actions

1. Review new structure (browse `core/` and `papers/`)
2. Test your workflow (run your usual commands)
3. Read MIGRATION_GUIDE.md for details
4. Start using `papers/radarcon2025_alss/` for your paper

---

**Status:** ✅ Migration Complete & Validated  
**Date:** November 4, 2025
