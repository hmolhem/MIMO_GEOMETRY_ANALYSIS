# Migration Commit Script
# Executes 5 organized commits for the paper-based structure migration

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     COMMITTING MIGRATION TO PAPER-BASED STRUCTURE       ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Commit 1: .gitignore
Write-Host "[1/5] Committing .gitignore updates..." -ForegroundColor Yellow
git add .gitignore
git commit -m "chore: update .gitignore for paper-based structure

- Change mimo-geom-dev/ to envs/
- Add datasets/ (large experimental data)
- Add papers/*/outputs/ and papers/*/figs/ (generated files)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit 1 complete" -ForegroundColor Green
} else {
    Write-Host "❌ Commit 1 failed" -ForegroundColor Red
    exit 1
}

# Commit 2: Core framework
Write-Host "`n[2/5] Committing core.radarpy package..." -ForegroundColor Yellow
git add core/radarpy/
git commit -m "feat: add core.radarpy package with modular structure

Core package organization:
- geometry: Array processors (ULA, Nested, Z1-Z6)
- algorithms: MUSIC variants, CRB, coarray utilities (merged from util/)
- signal: Simulation and array manifold (from sim/)
- metrics: Performance evaluation metrics
- io: Input/output utilities

All imports updated to new paths."

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit 2 complete" -ForegroundColor Green
} else {
    Write-Host "❌ Commit 2 failed" -ForegroundColor Red
    exit 1
}

# Commit 3: Scripts and tests
Write-Host "`n[3/5] Committing analysis scripts and tests..." -ForegroundColor Yellow
git add core/analysis_scripts/ core/tests/
git commit -m "feat: migrate analysis scripts and tests to core/

- Move all demo scripts with updated imports
- Update unit tests for new package structure
- All scripts validated and working"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit 3 complete" -ForegroundColor Green
} else {
    Write-Host "❌ Commit 3 failed" -ForegroundColor Red
    exit 1
}

# Commit 4: Papers
Write-Host "`n[4/5] Committing paper organization system..." -ForegroundColor Yellow
git add papers/
git commit -m "feat: add paper-based organization system

- papers/_template/: Reusable structure for new papers
- papers/radarcon2025_alss/: Current paper with all results
- Each paper isolated with own configs, scripts, outputs, figures"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit 4 complete" -ForegroundColor Green
} else {
    Write-Host "❌ Commit 4 failed" -ForegroundColor Red
    exit 1
}

# Commit 5: Documentation
Write-Host "`n[5/5] Committing migration documentation..." -ForegroundColor Yellow
git add MIGRATION_GUIDE.md QUICK_REFERENCE.md README_NEW.md RESTRUCTURE_COMPLETE.md COMMIT_MIGRATION.md
git commit -m "docs: add comprehensive migration documentation

- MIGRATION_GUIDE.md: Complete reference with rollback plan
- QUICK_REFERENCE.md: Quick commands for daily use
- README_NEW.md: Updated framework documentation
- RESTRUCTURE_COMPLETE.md: Migration success summary
- COMMIT_MIGRATION.md: Commit plan and verification steps"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Commit 5 complete" -ForegroundColor Green
} else {
    Write-Host "❌ Commit 5 failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           ALL MIGRATION COMMITS COMPLETE! ✅             ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Host "📝 Commit Summary:" -ForegroundColor Cyan
git log --oneline -5

Write-Host "`n🚀 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review commits: git log --oneline -5" -ForegroundColor White
Write-Host "  2. Push to remote: git push origin master" -ForegroundColor White
Write-Host "  3. Tag release: git tag -a v2.0.0 -m 'Paper-based structure migration'" -ForegroundColor White
Write-Host "  4. Push tags: git push origin --tags`n" -ForegroundColor White
