# Commit Plan: Migration to Paper-Based Structure

## Changes Summary

This migration restructures the project from a flat architecture to a paper-based research framework.

### Modified Files
- `.gitignore` - Updated for new structure (envs/, datasets/, papers/*/outputs/)

### New Files & Directories

#### 1. Core Framework (`core/`)
- `core/radarpy/` - Main Python package
  - `geometry/` - Array processors (ULA, Nested, Z1-Z6)
  - `algorithms/` - DOA algorithms (Spatial MUSIC, Coarray MUSIC, CRB, coarray utilities)
  - `signal/` - Signal simulation and array manifold
  - `metrics/` - Performance metrics
  - `io/` - I/O utilities
- `core/analysis_scripts/` - CLI demos and benchmarks
- `core/tests/` - Unit tests

#### 2. Paper Organization (`papers/`)
- `papers/_template/` - Reusable paper template
- `papers/radarcon2025_alss/` - Current paper with results

#### 3. Documentation
- `MIGRATION_GUIDE.md` - Complete migration reference
- `QUICK_REFERENCE.md` - Quick commands
- `README_NEW.md` - Updated framework documentation
- `RESTRUCTURE_COMPLETE.md` - Migration summary

## Staged Commits

### Commit 1: Update .gitignore for new structure
```bash
git add .gitignore
git commit -m "chore: update .gitignore for paper-based structure

- Change mimo-geom-dev/ to envs/
- Add datasets/ (large experimental data)
- Add papers/*/outputs/ and papers/*/figs/ (generated files)"
```

### Commit 2: Add core framework structure
```bash
git add core/radarpy/
git commit -m "feat: add core.radarpy package with modular structure

Core package organization:
- geometry: Array processors (ULA, Nested, Z1-Z6)
- algorithms: MUSIC variants, CRB, coarray utilities (merged from util/)
- signal: Simulation and array manifold (from sim/)
- metrics: Performance evaluation metrics
- io: Input/output utilities

All imports updated to new paths."
```

### Commit 3: Add analysis scripts and tests
```bash
git add core/analysis_scripts/ core/tests/
git commit -m "feat: migrate analysis scripts and tests to core/

- Move all demo scripts with updated imports
- Update unit tests for new package structure
- All scripts validated and working"
```

### Commit 4: Add paper organization system
```bash
git add papers/
git commit -m "feat: add paper-based organization system

- papers/_template/: Reusable structure for new papers
- papers/radarcon2025_alss/: Current paper with all results
- Each paper isolated with own configs, scripts, outputs, figures"
```

### Commit 5: Add migration documentation
```bash
git add MIGRATION_GUIDE.md QUICK_REFERENCE.md README_NEW.md RESTRUCTURE_COMPLETE.md
git commit -m "docs: add comprehensive migration documentation

- MIGRATION_GUIDE.md: Complete reference with rollback plan
- QUICK_REFERENCE.md: Quick commands for daily use
- README_NEW.md: Updated framework documentation
- RESTRUCTURE_COMPLETE.md: Migration success summary"
```

## Automated Commit Script

Save as `commit_migration.ps1`:

```powershell
# Commit 1: .gitignore
git add .gitignore
git commit -m "chore: update .gitignore for paper-based structure

- Change mimo-geom-dev/ to envs/
- Add datasets/ (large experimental data)
- Add papers/*/outputs/ and papers/*/figs/ (generated files)"

# Commit 2: Core framework
git add core/radarpy/
git commit -m "feat: add core.radarpy package with modular structure

Core package organization:
- geometry: Array processors (ULA, Nested, Z1-Z6)
- algorithms: MUSIC variants, CRB, coarray utilities (merged from util/)
- signal: Simulation and array manifold (from sim/)
- metrics: Performance evaluation metrics
- io: Input/output utilities

All imports updated to new paths."

# Commit 3: Scripts and tests
git add core/analysis_scripts/ core/tests/
git commit -m "feat: migrate analysis scripts and tests to core/

- Move all demo scripts with updated imports
- Update unit tests for new package structure
- All scripts validated and working"

# Commit 4: Papers
git add papers/
git commit -m "feat: add paper-based organization system

- papers/_template/: Reusable structure for new papers
- papers/radarcon2025_alss/: Current paper with all results
- Each paper isolated with own configs, scripts, outputs, figures"

# Commit 5: Documentation
git add MIGRATION_GUIDE.md QUICK_REFERENCE.md README_NEW.md RESTRUCTURE_COMPLETE.md
git commit -m "docs: add comprehensive migration documentation

- MIGRATION_GUIDE.md: Complete reference with rollback plan
- QUICK_REFERENCE.md: Quick commands for daily use
- README_NEW.md: Updated framework documentation
- RESTRUCTURE_COMPLETE.md: Migration success summary"

Write-Host "`n✅ All migration commits complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  git log --oneline -5" -ForegroundColor White
Write-Host "  git push origin master" -ForegroundColor White
```

## Quick Commit (All at Once)

If you prefer a single commit:

```bash
git add .
git commit -m "refactor: migrate to paper-based structure

Major restructuring:
- Reorganize into core/radarpy/ package structure
- Add papers/ organization system with template
- Update all imports and paths
- Relocate virtual env to envs/
- Add comprehensive documentation

All tests passing, functionality validated."
```

## Verification

After committing, verify:

```powershell
# Check commits
git log --oneline -5

# Verify tests still pass
.\envs\mimo-geom-dev\Scripts\Activate.ps1
python core\tests\test_coarray_mv.py

# Verify demos work
python core\analysis_scripts\run_z4_demo.py --N 7
```

## Notes

- Original files (`geometry_processors/`, `algorithms/`, etc.) still exist but are not tracked in these commits
- Can be removed after validation period
- Virtual environment relocated to `envs/` but gitignored
- All functionality tested and working before commit
