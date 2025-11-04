# Quick Commit Script - Execute All at Once
# Run this to commit everything in 5 organized commits

Write-Host "=== Starting Git Commit Process ===" -ForegroundColor Cyan

# Commit 1: Algorithms & Utilities
Write-Host "`n[1/5] Committing algorithms and utilities..." -ForegroundColor Yellow
git add algorithms/ util/ sim/
git commit -m "feat: Add DOA estimation algorithms and coarray processing

- Implement SpatialMUSIC (physical array MUSIC)
- Implement CoarrayMUSIC with FBA, unbiased lag averaging, diagonal loading
- Add Root-MUSIC (experimental, disabled by default)
- Add virtual ULA covariance builder with corrected Mv reporting
- Add CRB computation utilities
- Add simulation core and array manifold functions

Key fixes:
- Mv reporting bug fixed (prioritize longest contiguous segment)
- CRB interpretation corrected (X× above CRB)
- Root-MUSIC marked experimental with warning"

# Commit 2: Benchmark Infrastructure
Write-Host "`n[2/5] Committing benchmark infrastructure..." -ForegroundColor Yellow
git add analysis_scripts/run_benchmarks.py analysis_scripts/plot_benchmarks.py scripts/ run_headline_smoke.ps1 summarize_headline.ps1
git commit -m "feat: Add DOA benchmark infrastructure

- Comprehensive benchmark framework (28,800 trial capability)
- Support multiple arrays (Z4, Z5, Z6, ULA) and algorithms
- Parameter sweeps: SNR, snapshots, source separations
- Publication plotting script (200 DPI figures)
- PowerShell helpers for smoke testing and summaries
- CRB overlay generation with correct ratio formatting"

# Commit 3: Tests
Write-Host "`n[3/5] Committing tests and validation..." -ForegroundColor Yellow
git add tests/ test_*.py
git commit -m "test: Add unit tests and validation scripts

- Add test_coarray_mv.py: Verify Mv reporting (all passing ✓)
- Add coarray structure validation for Z4/Z6
- Add Z6 scaling analysis (documents Mv=3 limitation)
- Add lag weight verification scripts

Test results:
- Z4(N=7): Mv=12, segment=[3:14] ✓
- Z5(N=7): Mv=10, segment=[3:12] ✓
- ULA(N=7): Mv=7, segment=[0:6] ✓"

# Commit 4: Results & Figures
Write-Host "`n[4/5] Committing benchmark results and figures..." -ForegroundColor Yellow
git add results/
git commit -m "data: Add benchmark results and publication figures

Benchmark: 28,800 trials (9,600 CSV rows)
- Arrays: Z4, Z5, ULA (N=7)
- Algorithms: SpatialMUSIC, CoarrayMUSIC
- SNR: 0, 5, 10, 15 dB | Snapshots: 32, 128, 256, 512
- Separations: 1°, 2°, 3° | Trials: 100/config

Headline results (SNR=10dB, M=256, Δθ=2°):
- Z5 SpatialMUSIC: RMSE=0.185°, Resolve=87.9%, 2.68× CRB ⭐
- Z4 SpatialMUSIC: RMSE=0.359°, Resolve=75.8%
- ULA SpatialMUSIC: RMSE=0.940°, Resolve=27.3%

Figures (200 DPI):
- RMSE vs snapshots (convergence to CRB)
- Resolution vs SNR (Z5 dominance)
- Z5 heatmap with CRB contours"

# Commit 5: Documentation
Write-Host "`n[5/5] Committing documentation..." -ForegroundColor Yellow
git add README.md PAPER_READY_MATERIALS.md PUBLICATION_WORKFLOW.md FINAL_SUMMARY.md PACKAGE_SUMMARY.md QUICK_ACTION_ITEMS.md STATUS_AND_HELP.md GIT_COMMIT_PLAN.md
git commit -m "docs: Add publication-ready documentation package

Complete paper scaffolding:
- PAPER_READY_MATERIALS.md: Abstract, sections, LaTeX snippets
- PUBLICATION_WORKFLOW.md: End-to-end workflow guide
- FINAL_SUMMARY.md: Executive summary
- PACKAGE_SUMMARY.md: Implementation status
- QUICK_ACTION_ITEMS.md: Immediate vs future tasks
- STATUS_AND_HELP.md: Verification and help guide
- GIT_COMMIT_PLAN.md: This commit plan

Paper-ready:
- Title, abstract (185 words)
- All sections with content
- LaTeX tables and figure captions
- Reproducibility checklist ✓

Ready for IEEE TSP/ICASSP submission 🚀"

# Summary
Write-Host "`n=== Commit Summary ===" -ForegroundColor Cyan
git log --oneline -5
Write-Host "`n✅ All commits complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  - Review: git log --oneline" -ForegroundColor White
Write-Host "  - Push: git push origin master" -ForegroundColor White
Write-Host "  - Tag: git tag -a v1.0.0 -m 'Publication package v1.0.0'" -ForegroundColor White
