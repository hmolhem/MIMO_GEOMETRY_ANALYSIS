# Git Commit Plan - Publication Package

## 📋 Commit Strategy (Organized by Function)

### Commit 1: Core DOA Algorithms & Utilities
**Purpose**: Add fundamental DOA estimation algorithms and coarray processing

```powershell
# Add algorithm implementations
git add algorithms/spatial_music.py
git add algorithms/coarray_music.py
git add algorithms/__init__.py

# Add utility functions
git add util/coarray.py
git add util/crb.py

# Add simulation infrastructure
git add sim/doa_sim_core.py
git add sim/array_manifold.py
git add sim/metrics.py

# Commit
git commit -m "feat: Add DOA estimation algorithms and coarray processing

- Implement SpatialMUSIC (physical array MUSIC)
- Implement CoarrayMUSIC with FBA, unbiased lag averaging, diagonal loading
- Add Root-MUSIC (experimental, disabled by default)
- Add virtual ULA covariance builder with corrected Mv reporting
- Add CRB computation utilities
- Add simulation core and array manifold functions
- Add DOA estimation metrics (RMSE, resolution)

Key fixes:
- Mv reporting bug fixed (prioritize longest contiguous segment)
- CRB interpretation corrected ('X× above CRB' instead of 'better than')
- Root-MUSIC marked experimental with warning"
```

---

### Commit 2: Benchmark Infrastructure
**Purpose**: Add comprehensive benchmark framework

```powershell
# Add benchmark scripts
git add analysis_scripts/run_benchmarks.py
git add analysis_scripts/plot_benchmarks.py

# Add plotting and analysis
git add scripts/plot_headline.py

# Add PowerShell helpers
git add run_headline_smoke.ps1
git add summarize_headline.ps1

# Commit
git commit -m "feat: Add DOA benchmark infrastructure

- Implement comprehensive benchmark framework (run_benchmarks.py)
- Support multiple arrays (Z4, Z5, Z6, ULA), algorithms (Spatial/Coarray MUSIC)
- Parameter sweeps: SNR, snapshots, source separations
- CRB overlay generation for performance comparison
- Publication-ready plotting script (plot_headline.py)
- PowerShell helpers for smoke testing and result summaries

Features:
- 28,800 trial benchmark capability
- Fixed seed for reproducibility
- 21-column CSV output with geometry metrics
- Automatic figure generation (200 DPI PNG)
- CRB ratio formatting with correct interpretation"
```

---

### Commit 3: Test Suite & Validation
**Purpose**: Add unit tests and validation scripts

```powershell
# Add formal tests
git add tests/test_coarray_mv.py

# Add debugging/validation scripts
git add test_z4_coarray.py
git add test_z4_lags.py
git add test_z6_coarray.py
git add test_z6_scaling.py
git add test_lag_weights.py

# Commit
git commit -m "test: Add unit tests and validation scripts

- Add test_coarray_mv.py: Verify Mv reporting for Z4/Z5/ULA
- Add validation scripts for Z4/Z6 coarray structures
- Add Z6 scaling analysis (documents Mv=3 limitation)
- Add lag weight verification

Test coverage:
- Mv computation correctness
- Contiguous segment detection
- Coarray fragmentation analysis
- All tests passing ✓"
```

---

### Commit 4: Benchmark Results & Figures
**Purpose**: Add generated benchmark data and publication figures

```powershell
# Add benchmark results
git add results/bench/headline.csv
git add results/bench/crb_overlay.csv

# Add publication figures
git add results/figs/rmse_vs_M_SNR10_delta2.png
git add results/figs/resolve_vs_SNR_M256_delta2.png
git add results/figs/heatmap_Z5_spatial.png
git add results/figs/headline_table_SNR10_M256_delta2.csv

# Commit
git commit -m "data: Add benchmark results and publication figures

Benchmark configuration:
- Arrays: Z4, Z5, ULA (N=7)
- Algorithms: SpatialMUSIC, CoarrayMUSIC
- SNR: 0, 5, 10, 15 dB
- Snapshots: 32, 128, 256, 512
- Separations: 1°, 2°, 3°
- Trials: 100 per configuration
- Total: 28,800 runs (9,600 CSV rows)

Key results (SNR=10dB, M=256, Δθ=2°):
- Z5 SpatialMUSIC: RMSE=0.185°, Resolve=87.9%, 2.68× CRB
- Z4 SpatialMUSIC: RMSE=0.359°, Resolve=75.8%
- ULA SpatialMUSIC: RMSE=0.940°, Resolve=27.3%

Figures:
- RMSE vs snapshots (showing convergence to CRB)
- Resolution vs SNR (Z5 dominance)
- Z5 heatmap with CRB contours"
```

---

### Commit 5: Documentation & Publication Materials
**Purpose**: Add complete publication-ready documentation

```powershell
# Add documentation
git add README.md
git add PAPER_READY_MATERIALS.md
git add PUBLICATION_WORKFLOW.md
git add FINAL_SUMMARY.md
git add PACKAGE_SUMMARY.md
git add QUICK_ACTION_ITEMS.md
git add STATUS_AND_HELP.md

# Commit
git commit -m "docs: Add publication-ready documentation package

Complete documentation suite:
- README.md: Updated with overview and limitations
- PAPER_READY_MATERIALS.md: Full paper scaffolding (abstract, sections, LaTeX)
- PUBLICATION_WORKFLOW.md: End-to-end workflow guide
- FINAL_SUMMARY.md: Executive summary of deliverables
- PACKAGE_SUMMARY.md: Implementation status and known issues
- QUICK_ACTION_ITEMS.md: Immediate vs future tasks
- STATUS_AND_HELP.md: Verification checklist and help guide

Paper-ready materials:
- Title, abstract (185 words)
- Introduction, methods, results, discussion, conclusion
- LaTeX tables and figure captions (drop-in ready)
- Reproducibility checklist

Key findings documented:
- Z5 near-CRB performance (2.68× bound)
- Z6 coarray fragmentation (Mv=3 design limitation)
- SpatialMUSIC > CoarrayMUSIC in this regime
- Root-MUSIC experimental status"
```

---

## 🚀 Execute All Commits (One Command)

```powershell
# Execute all 5 commits in sequence
git add algorithms/ util/ sim/
git commit -m "feat: Add DOA estimation algorithms and coarray processing

- Implement SpatialMUSIC and CoarrayMUSIC with FBA/diagonal loading
- Add virtual ULA covariance builder with corrected Mv reporting
- Add CRB computation and simulation utilities
- Fix Mv reporting bug, correct CRB interpretation
- Mark Root-MUSIC as experimental"

git add analysis_scripts/run_benchmarks.py analysis_scripts/plot_benchmarks.py scripts/ run_headline_smoke.ps1 summarize_headline.ps1
git commit -m "feat: Add DOA benchmark infrastructure

- Comprehensive benchmark framework (28,800 trial capability)
- Publication plotting script (200 DPI figures)
- PowerShell helpers for smoke testing and summaries
- CRB overlay generation and ratio formatting"

git add tests/ test_*.py
git commit -m "test: Add unit tests and validation scripts

- Verify Mv reporting for Z4/Z5/ULA (all passing)
- Add coarray structure validation
- Add Z6 scaling analysis (documents Mv=3 limitation)"

git add results/
git commit -m "data: Add benchmark results and publication figures

Results: 28,800 trials, Z5 winner (RMSE=0.185°, 87.9% resolve, 2.68× CRB)
Figures: RMSE vs M, Resolve vs SNR, Z5 heatmap with CRB contours"

git add *.md README.md
git commit -m "docs: Add publication-ready documentation package

Complete paper scaffolding, LaTeX snippets, workflow guide, and executive summary
Ready for IEEE TSP/ICASSP submission"
```

---

## ✅ Verification Checklist

After committing, verify:

```powershell
# Check all commits
git log --oneline -5

# Verify nothing left uncommitted
git status

# See commit details
git show --stat HEAD
git show --stat HEAD~1
git show --stat HEAD~2
git show --stat HEAD~3
git show --stat HEAD~4
```

---

## 🔍 Optional: Create .gitignore

If you want to exclude certain files:

```powershell
# Create .gitignore
@"
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg-info/

# Virtual environment
mimo-geom-dev/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Large data files (optional - you may want to commit these)
# results/bench/*.csv
# results/figs/*.png
"@ | Out-File -FilePath .gitignore -Encoding utf8

git add .gitignore
git commit -m "chore: Add .gitignore"
```

---

## 📤 Push to Remote (After Commits)

```powershell
# Push to GitHub/GitLab
git push origin master

# Or if you want to create a release tag
git tag -a v1.0.0 -m "Publication package v1.0.0 - Z5 near-CRB performance"
git push origin v1.0.0
```

---

## 🎯 Summary

**5 logical commits organized by function:**
1. Core algorithms & utilities
2. Benchmark infrastructure
3. Test suite
4. Results & figures
5. Documentation

**Total files being committed:**
- 10+ source code files (algorithms, utils, sim)
- 3 benchmark scripts + 2 PowerShell helpers
- 5+ test files
- 2 CSV files + 4 figure files
- 7 documentation files

**Ready to execute!** Use the commands above to commit everything in an organized way. 🚀
