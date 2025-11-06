# ============================================================================
# Cleanup Script - Move Files to Garbage Folder
# ============================================================================
# Moves duplicate and temporary files to a 'garbage' folder for safekeeping

$projectRoot = $PSScriptRoot
$garbageFolder = Join-Path $projectRoot "garbage"

# Create garbage folder if it doesn't exist
if (-not (Test-Path $garbageFolder)) {
    New-Item -ItemType Directory -Path $garbageFolder -Force | Out-Null
    Write-Host "[CREATED] garbage/ folder" -ForegroundColor Cyan
}

# Track statistics
$script:filesMoved = 0
$script:directoriesMoved = 0
$script:bytesMoved = 0

# Function to safely move items to garbage folder
function Move-ToGarbage {
    param(
        [string]$Path,
        [string]$Description
    )
    
    $fullPath = Join-Path $projectRoot $Path
    
    if (Test-Path $fullPath) {
        try {
            # Get item info for statistics
            $item = Get-Item $fullPath
            
            # Calculate size
            if ($item.PSIsContainer) {
                $size = (Get-ChildItem $fullPath -Recurse -File | Measure-Object -Property Length -Sum).Sum
                if ($null -eq $size) { $size = 0 }
            } else {
                $size = $item.Length
            }
            
            # Create destination path in garbage folder
            $destPath = Join-Path $garbageFolder $Path
            $destDir = Split-Path $destPath -Parent
            
            # Create destination directory if needed
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            
            # Move the item
            Move-Item -Path $fullPath -Destination $destPath -Force
            
            # Update statistics
            if ($item.PSIsContainer) {
                $script:directoriesMoved++
            } else {
                $script:filesMoved++
            }
            $script:bytesMoved += $size
            
            Write-Host "[MOVED] $Description" -ForegroundColor Green
            
        } catch {
            Write-Host "[FAILED] $Description - $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "[SKIP] Not found: $Description" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Moving Files to Garbage Folder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Category 1: Duplicate Markdown Documentation Files
# ============================================================================
Write-Host "Category 1: Duplicate Markdown Files" -ForegroundColor Yellow
Move-ToGarbage "README_NEW.md" "Old README file"
Move-ToGarbage "FINAL_SUMMARY.md" "Old summary file"
Move-ToGarbage "PACKAGE_SUMMARY.md" "Old package summary"
Move-ToGarbage "COMMIT_MIGRATION.md" "Old migration doc"
Move-ToGarbage "GIT_COMMIT_PLAN.md" "Old git plan"
Move-ToGarbage "MIGRATION_GUIDE.md" "Old migration guide"
Move-ToGarbage "PAPER_READY_MATERIALS.md" "Old paper materials"
Move-ToGarbage "PROJECT_STRUCTURE.md" "Old project structure"
Move-ToGarbage "PUBLICATION_WORKFLOW.md" "Old publication workflow"
Move-ToGarbage "QUICK_ACTION_ITEMS.md" "Old action items"
Move-ToGarbage "QUICK_REFERENCE.md" "Old quick reference"
Move-ToGarbage "RESTRUCTURE_COMPLETE.md" "Old restructure doc"
Move-ToGarbage "STATUS_AND_HELP.md" "Old status doc"

# ============================================================================
# Category 2: Duplicate Code Directories (superseded by core/radarpy/)
# ============================================================================
Write-Host ""
Write-Host "Category 2: Duplicate Code Directories" -ForegroundColor Yellow
Move-ToGarbage "algorithms" "Duplicate algorithms directory"
Move-ToGarbage "sim" "Duplicate sim directory"
Move-ToGarbage "util" "Duplicate util directory"

# ============================================================================
# Category 3: Standalone Test Files (consolidated into tests/)
# ============================================================================
Write-Host ""
Write-Host "Category 3: Standalone Test Files" -ForegroundColor Yellow
Move-ToGarbage "test_lag_weights.py" "Standalone test file"
Move-ToGarbage "test_z4_coarray.py" "Standalone Z4 test"
Move-ToGarbage "test_z4_lags.py" "Standalone Z4 lags test"
Move-ToGarbage "test_z6_coarray.py" "Standalone Z6 test"
Move-ToGarbage "test_z6_scaling.py" "Standalone Z6 scaling test"
Move-ToGarbage "test" "Old test directory"

# ============================================================================
# Category 4: Export Archive
# ============================================================================
Write-Host ""
Write-Host "Category 4: Export Archive" -ForegroundColor Yellow
Move-ToGarbage "export" "Export directory"
Move-ToGarbage "MIMO_GEOMETRY_ANALYSIS_export.zip" "Export ZIP file"

# ============================================================================
# Category 5: Temporary Generated Files
# ============================================================================
Write-Host ""
Write-Host "Category 5: Temporary Files" -ForegroundColor Yellow
Move-ToGarbage "simple_plot.png" "Temporary plot"
Move-ToGarbage "project_structure.txt" "Old structure file"
Move-ToGarbage "project_tree.txt" "Old tree file"

# ============================================================================
# Category 6: Temporary PowerShell Scripts
# ============================================================================
Write-Host ""
Write-Host "Category 6: Temporary Scripts" -ForegroundColor Yellow
Move-ToGarbage "commit_all.ps1" "Old commit script"
Move-ToGarbage "commit_migration.ps1" "Old migration commit script"
Move-ToGarbage "run_headline_smoke.ps1" "Old smoke test script"
Move-ToGarbage "summarize_headline.ps1" "Old summarize script"

# ============================================================================
# Category 7: Python Cache Directories
# ============================================================================
Write-Host ""
Write-Host "Category 7: Python Cache" -ForegroundColor Yellow

$pycacheDirs = Get-ChildItem -Path $projectRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($dir in $pycacheDirs) {
    $relativePath = $dir.FullName.Replace($projectRoot, "").TrimStart("\")
    Move-ToGarbage $relativePath "Python cache: $relativePath"
}

# ============================================================================
# Category 8: Compiled Python Files
# ============================================================================
Write-Host ""
Write-Host "Category 8: Compiled Python Files" -ForegroundColor Yellow

$pycFiles = Get-ChildItem -Path $projectRoot -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue
foreach ($file in $pycFiles) {
    $relativePath = $file.FullName.Replace($projectRoot, "").TrimStart("\")
    Move-ToGarbage $relativePath "Compiled Python file"
}

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Move Summary:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Files moved: $script:filesMoved" -ForegroundColor Green
Write-Host "Directories moved: $script:directoriesMoved" -ForegroundColor Green
Write-Host "Space moved: $([math]::Round($script:bytesMoved/1MB, 2)) MB" -ForegroundColor Green
Write-Host ""
Write-Host "All files are now in the 'garbage/' folder." -ForegroundColor Green
Write-Host "You can review and permanently delete them later if needed." -ForegroundColor Yellow
Write-Host ""
