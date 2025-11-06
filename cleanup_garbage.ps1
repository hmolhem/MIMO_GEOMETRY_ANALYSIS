# cleanup_garbage.ps1
# Remove unnecessary/duplicate/garbage files from MIMO_GEOMETRY_ANALYSIS project
# Date: November 6, 2025

Write-Host "=== MIMO Array Geometry Analysis - Cleanup Script ===" -ForegroundColor Cyan
Write-Host ""

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

# Counters
$filesRemoved = 0
$dirsRemoved = 0
$bytesFreed = 0

function Remove-ItemSafe {
    param($Path, $Description)
    if (Test-Path $Path) {
        try {
            $size = (Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            Remove-Item $Path -Recurse -Force -ErrorAction Stop
            Write-Host "✓ Removed: $Description" -ForegroundColor Green
            Write-Host "  Path: $Path" -ForegroundColor Gray
            if ($size) {
                $script:bytesFreed += $size
                Write-Host "  Size: $([math]::Round($size/1MB, 2)) MB" -ForegroundColor Gray
            }
            if (Test-Path $Path -PathType Container) {
                $script:dirsRemoved++
            } else {
                $script:filesRemoved++
            }
            return $true
        } catch {
            Write-Host "✗ Failed to remove: $Description" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "○ Not found (already removed?): $Description" -ForegroundColor Yellow
        return $false
    }
}

Write-Host "Step 1: Removing duplicate markdown documentation files..." -ForegroundColor Yellow
Write-Host ""

# Duplicate/old README files
Remove-ItemSafe "README_NEW.md" "Old README_NEW.md (superseded by README.md)"

# Temporary summary files (info already in docs/)
Remove-ItemSafe "FINAL_SUMMARY.md" "Final Summary (temporary file)"
Remove-ItemSafe "PACKAGE_SUMMARY.md" "Package Summary (temporary file)"
Remove-ItemSafe "COMMIT_MIGRATION.md" "Commit Migration doc (temporary)"
Remove-ItemSafe "GIT_COMMIT_PLAN.md" "Git Commit Plan (temporary)"
Remove-ItemSafe "MIGRATION_GUIDE.md" "Migration Guide (temporary)"
Remove-ItemSafe "PAPER_READY_MATERIALS.md" "Paper Ready Materials (temporary)"
Remove-ItemSafe "PROJECT_STRUCTURE.md" "Project Structure (duplicate)"
Remove-ItemSafe "PUBLICATION_WORKFLOW.md" "Publication Workflow (temporary)"
Remove-ItemSafe "QUICK_ACTION_ITEMS.md" "Quick Action Items (temporary)"
Remove-ItemSafe "QUICK_REFERENCE.md" "Quick Reference (temporary)"
Remove-ItemSafe "RESTRUCTURE_COMPLETE.md" "Restructure Complete (temporary)"
Remove-ItemSafe "STATUS_AND_HELP.md" "Status and Help (temporary)"

Write-Host ""
Write-Host "Step 2: Removing duplicate/old code directories..." -ForegroundColor Yellow
Write-Host ""

# Duplicate algorithms directory (core/radarpy/algorithms/ is the real one)
Remove-ItemSafe "algorithms" "Old algorithms/ directory (use core/radarpy/algorithms/)"

# Duplicate sim directory (core/radarpy/signal/ is the real one)
Remove-ItemSafe "sim" "Old sim/ directory (use core/radarpy/signal/)"

# Duplicate util directory (functions migrated to core/)
Remove-ItemSafe "util" "Old util/ directory (migrated to core/)"

Write-Host ""
Write-Host "Step 3: Removing standalone test files (use core/tests/ or tests/)..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "test_lag_weights.py" "Standalone test file"
Remove-ItemSafe "test_z4_coarray.py" "Standalone test file"
Remove-ItemSafe "test_z4_lags.py" "Standalone test file"
Remove-ItemSafe "test_z6_coarray.py" "Standalone test file"
Remove-ItemSafe "test_z6_scaling.py" "Standalone test file"

# Duplicate test directory (consolidate to core/tests/ or tests/)
Remove-ItemSafe "test" "Old test/ directory (use tests/ or core/tests/)"

Write-Host ""
Write-Host "Step 4: Removing export archive (already distributed)..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "export" "Export directory (archived content)"
Remove-ItemSafe "MIMO_GEOMETRY_ANALYSIS_export.zip" "Export ZIP file"

Write-Host ""
Write-Host "Step 5: Removing temporary/generated files..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "simple_plot.png" "Temporary plot file"
Remove-ItemSafe "project_structure.txt" "Generated project structure file"
Remove-ItemSafe "project_tree.txt" "Generated project tree file"

Write-Host ""
Write-Host "Step 6: Removing temporary PowerShell scripts..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "commit_all.ps1" "Temporary commit script"
Remove-ItemSafe "commit_migration.ps1" "Temporary migration script"
Remove-ItemSafe "run_headline_smoke.ps1" "Temporary smoke test script"
Remove-ItemSafe "summarize_headline.ps1" "Temporary summarize script"

Write-Host ""
Write-Host "Step 7: Cleaning __pycache__ directories..." -ForegroundColor Yellow
Write-Host ""

$pycacheDirs = Get-ChildItem -Path $projectRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($dir in $pycacheDirs) {
    Remove-ItemSafe $dir.FullName "Python cache: $($dir.FullName.Replace($projectRoot, '.'))"
}

Write-Host ""
Write-Host "Step 8: Cleaning .pyc files..." -ForegroundColor Yellow
Write-Host ""

$pycFiles = Get-ChildItem -Path $projectRoot -Recurse -Filter "*.pyc" -File -ErrorAction SilentlyContinue
foreach ($file in $pycFiles) {
    Remove-ItemSafe $file.FullName "Compiled Python: $($file.FullName.Replace($projectRoot, '.'))"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cleanup Summary:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Files removed: $filesRemoved" -ForegroundColor Green
Write-Host "Directories removed: $dirsRemoved" -ForegroundColor Green
Write-Host "Space freed: $([math]::Round($bytesFreed/1MB, 2)) MB" -ForegroundColor Green
Write-Host ""

# Display remaining project structure
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Current Clean Project Structure:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "MIMO_GEOMETRY_ANALYSIS/" -ForegroundColor White
Write-Host "├── README.md                   # Main documentation" -ForegroundColor Gray
Write-Host "├── requirements.txt            # Python dependencies" -ForegroundColor Gray
Write-Host "├── DOCUMENTATION_COMPLETE.md   # Documentation report" -ForegroundColor Gray
Write-Host "├── .gitignore                  # Git ignore rules" -ForegroundColor Gray
Write-Host "├── .github/                    # GitHub config" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── docs/                       # Documentation" -ForegroundColor White
Write-Host "│   ├── GETTING_STARTED.md" -ForegroundColor Gray
Write-Host "│   ├── API_REFERENCE.md" -ForegroundColor Gray
Write-Host "│   ├── DOCUMENTATION_UPDATE_SUMMARY.md" -ForegroundColor Gray
Write-Host "│   └── tutorials/" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── geometry_processors/        # Array geometry analysis" -ForegroundColor White
Write-Host "│   ├── bases_classes.py        # Core framework" -ForegroundColor Gray
Write-Host "│   ├── ula_processors*.py      # ULA arrays" -ForegroundColor Gray
Write-Host "│   ├── nested_processor*.py    # Nested arrays" -ForegroundColor Gray
Write-Host "│   └── z*_processor*.py        # Z1-Z6 arrays" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── core/                       # Core algorithms & utilities" -ForegroundColor White
Write-Host "│   ├── radarpy/                # Main package" -ForegroundColor Gray
Write-Host "│   │   ├── algorithms/         # DOA estimation (MUSIC, ALSS)" -ForegroundColor Gray
Write-Host "│   │   ├── signal/             # Signal generation & coarray" -ForegroundColor Gray
Write-Host "│   │   ├── metrics/            # CRB, performance metrics" -ForegroundColor Gray
Write-Host "│   │   └── geometry/           # Array geometry (duplicate processors)" -ForegroundColor Gray
Write-Host "│   ├── analysis_scripts/       # Analysis demos & benchmarks" -ForegroundColor Gray
Write-Host "│   └── tests/                  # Unit tests" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── analysis_scripts/           # Interactive demos" -ForegroundColor White
Write-Host "│   ├── graphical_demo.py       # Menu-driven analysis" -ForegroundColor Gray
Write-Host "│   ├── methods_demo.py         # Method validation" -ForegroundColor Gray
Write-Host "│   └── run_*_demo*.py          # Array-specific demos" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── scripts/                    # Batch processing scripts" -ForegroundColor White
Write-Host "│   ├── run_paper_benchmarks.py # Paper-ready benchmarks" -ForegroundColor Gray
Write-Host "│   └── plot_headline.py        # Headline figures" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── tools/                      # Analysis utilities" -ForegroundColor White
Write-Host "│   ├── plot_paper_benchmarks.py" -ForegroundColor Gray
Write-Host "│   ├── analyze_svd.py" -ForegroundColor Gray
Write-Host "│   └── *.py                    # Various plotting tools" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── papers/                     # Publication materials" -ForegroundColor White
Write-Host "│   └── radarcon2025_alss/      # RadarCon 2025 paper" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── results/                    # Generated outputs" -ForegroundColor White
Write-Host "│   ├── bench/                  # Benchmark results" -ForegroundColor Gray
Write-Host "│   ├── plots/                  # Generated plots" -ForegroundColor Gray
Write-Host "│   ├── summaries/              # CSV summaries" -ForegroundColor Gray
Write-Host "│   └── svd/                    # SVD analysis" -ForegroundColor Gray
Write-Host "│" -ForegroundColor Gray
Write-Host "├── tests/                      # Additional unit tests" -ForegroundColor White
Write-Host "├── mimo-geom-dev/              # Python virtual environment" -ForegroundColor White
Write-Host "└── miniScript/                 # Quick test scripts" -ForegroundColor White
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cleanup complete! Project is now clean." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review remaining files" -ForegroundColor Gray
Write-Host "  2. Run tests: python -m pytest core/tests/" -ForegroundColor Gray
Write-Host "  3. Commit cleanup: git add -A && git commit -m ""chore: remove duplicate files""" -ForegroundColor Gray
Write-Host ""
