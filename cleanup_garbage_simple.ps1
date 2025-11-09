# cleanup_garbage_simple.ps1
# Remove unnecessary/duplicate/garbage files
# Date: November 6, 2025

Write-Host "=== MIMO Array Geometry Analysis - Cleanup Script ===" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "c:\MyDocument\RadarPy\MIMO_GEOMETRY_ANALYSIS"
Set-Location $projectRoot

$filesRemoved = 0
$dirsRemoved = 0
$bytesFreed = 0

function Remove-ItemSafe {
    param($Path, $Description)
    $fullPath = Join-Path $projectRoot $Path
    if (Test-Path $fullPath) {
        try {
            $size = 0
            if (Test-Path $fullPath -PathType Container) {
                $size = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            } else {
                $size = (Get-Item $fullPath).Length
            }
            
            Remove-Item $fullPath -Recurse -Force -ErrorAction Stop
            Write-Host "[REMOVED] $Description" -ForegroundColor Green
            Write-Host "  Path: $Path" -ForegroundColor Gray
            if ($size) {
                $sizeMB = [math]::Round($size/1MB, 2)
                $script:bytesFreed += $size
                Write-Host "  Size: $sizeMB MB" -ForegroundColor Gray
            }
            
            if (Test-Path $fullPath -PathType Container) {
                $script:dirsRemoved++
            } else {
                $script:filesRemoved++
            }
            return $true
        } catch {
            Write-Host "[FAILED] $Description" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "[SKIP] Not found: $Description" -ForegroundColor Yellow
        return $false
    }
}

Write-Host "Step 1: Removing duplicate markdown documentation files..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "README_NEW.md" "Old README_NEW.md"
Remove-ItemSafe "FINAL_SUMMARY.md" "Final Summary"
Remove-ItemSafe "PACKAGE_SUMMARY.md" "Package Summary"
Remove-ItemSafe "COMMIT_MIGRATION.md" "Commit Migration doc"
Remove-ItemSafe "GIT_COMMIT_PLAN.md" "Git Commit Plan"
Remove-ItemSafe "MIGRATION_GUIDE.md" "Migration Guide"
Remove-ItemSafe "PAPER_READY_MATERIALS.md" "Paper Ready Materials"
Remove-ItemSafe "PROJECT_STRUCTURE.md" "Project Structure"
Remove-ItemSafe "PUBLICATION_WORKFLOW.md" "Publication Workflow"
Remove-ItemSafe "QUICK_ACTION_ITEMS.md" "Quick Action Items"
Remove-ItemSafe "QUICK_REFERENCE.md" "Quick Reference"
Remove-ItemSafe "RESTRUCTURE_COMPLETE.md" "Restructure Complete"
Remove-ItemSafe "STATUS_AND_HELP.md" "Status and Help"

Write-Host ""
Write-Host "Step 2: Removing duplicate/old code directories..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "algorithms" "Old algorithms directory"
Remove-ItemSafe "sim" "Old sim directory"
Remove-ItemSafe "util" "Old util directory"

Write-Host ""
Write-Host "Step 3: Removing standalone test files..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "test_lag_weights.py" "Standalone test file"
Remove-ItemSafe "test_z4_coarray.py" "Standalone test file"
Remove-ItemSafe "test_z4_lags.py" "Standalone test file"
Remove-ItemSafe "test_z6_coarray.py" "Standalone test file"
Remove-ItemSafe "test_z6_scaling.py" "Standalone test file"
Remove-ItemSafe "test" "Old test directory"

Write-Host ""
Write-Host "Step 4: Removing export archive..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "export" "Export directory"
Remove-ItemSafe "MIMO_GEOMETRY_ANALYSIS_export.zip" "Export ZIP file"

Write-Host ""
Write-Host "Step 5: Removing temporary/generated files..." -ForegroundColor Yellow
Write-Host ""

Remove-ItemSafe "simple_plot.png" "Temporary plot file"
Remove-ItemSafe "project_structure.txt" "Generated project structure"
Remove-ItemSafe "project_tree.txt" "Generated project tree"

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
    $relativePath = $dir.FullName.Replace($projectRoot + "\", "")
    if (Remove-ItemSafe $relativePath "Python cache") {
        # Count handled by function
    }
}

Write-Host ""
Write-Host "Step 8: Cleaning .pyc files..." -ForegroundColor Yellow
Write-Host ""

$pycFiles = Get-ChildItem -Path $projectRoot -Recurse -Filter "*.pyc" -File -ErrorAction SilentlyContinue
foreach ($file in $pycFiles) {
    $relativePath = $file.FullName.Replace($projectRoot + "\", "")
    if (Remove-ItemSafe $relativePath "Compiled Python file") {
        # Count handled by function
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cleanup Summary:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Files removed: $filesRemoved" -ForegroundColor Green
Write-Host "Directories removed: $dirsRemoved" -ForegroundColor Green
$mbFreed = [math]::Round($bytesFreed/1MB, 2)
Write-Host "Space freed: $mbFreed MB" -ForegroundColor Green
Write-Host ""
Write-Host "Cleanup complete! Project is now clean." -ForegroundColor Green
Write-Host ""
