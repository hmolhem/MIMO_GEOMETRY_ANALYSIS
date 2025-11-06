# Simple Project Export Script
# Creates a ZIP file with code and documentation (excludes venv and large results)

Write-Host "`n=== MIMO GEOMETRY ANALYSIS PROJECT EXPORT ===`n" -ForegroundColor Cyan

# 1. Generate project tree
Write-Host "[1/4] Generating project structure..." -ForegroundColor Yellow
tree /F /A > export\project_tree.txt
Write-Host "  Done: export\project_tree.txt" -ForegroundColor Green

# 2. Create directories
Write-Host "`n[2/4] Setting up export structure..." -ForegroundColor Yellow
$dirs = @("export\core", "export\geometry_processors", "export\tools", "export\scripts", "export\papers")
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "  Done" -ForegroundColor Green

# 3. Copy files
Write-Host "`n[3/4] Copying files..." -ForegroundColor Yellow

# Copy core
Copy-Item -Path "core\*" -Destination "export\core\" -Recurse -Force -Exclude __pycache__,*.pyc
Write-Host "  Copied: core/" -ForegroundColor Gray

# Copy geometry processors
Copy-Item -Path "geometry_processors\*" -Destination "export\geometry_processors\" -Recurse -Force -Exclude __pycache__,*.pyc
Write-Host "  Copied: geometry_processors/" -ForegroundColor Gray

# Copy tools
Copy-Item -Path "tools\*" -Destination "export\tools\" -Recurse -Force
Write-Host "  Copied: tools/" -ForegroundColor Gray

# Copy scripts
Copy-Item -Path "scripts\*" -Destination "export\scripts\" -Recurse -Force
Write-Host "  Copied: scripts/" -ForegroundColor Gray

# Copy papers
Copy-Item -Path "papers\*" -Destination "export\papers\" -Recurse -Force
Write-Host "  Copied: papers/" -ForegroundColor Gray

# Copy root files
Copy-Item -Path "requirements.txt" -Destination "export\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "README.md" -Destination "export\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "PROJECT_STRUCTURE.md" -Destination "export\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path ".gitignore" -Destination "export\" -Force -ErrorAction SilentlyContinue

Write-Host "  Done" -ForegroundColor Green

# 4. Create ZIP
Write-Host "`n[4/4] Creating ZIP archive..." -ForegroundColor Yellow

$zipFile = "MIMO_GEOMETRY_ANALYSIS_export.zip"
if (Test-Path $zipFile) {
    Remove-Item $zipFile -Force
}

Compress-Archive -Path "export\*" -DestinationPath $zipFile -Force

$zipSize = [math]::Round((Get-Item $zipFile).Length / 1MB, 2)

Write-Host "  Done: $zipFile" -ForegroundColor Green

# Summary
Write-Host "`n=== EXPORT COMPLETE ===`n" -ForegroundColor Cyan
Write-Host "File: $zipFile" -ForegroundColor White
Write-Host "Size: $zipSize MB" -ForegroundColor White
Write-Host "`nContents:" -ForegroundColor Yellow
Write-Host "  - All Python source code" -ForegroundColor Gray
Write-Host "  - Documentation (MD files)" -ForegroundColor Gray
Write-Host "  - Analysis tools and scripts" -ForegroundColor Gray
Write-Host "  - Paper materials (guides, figures)" -ForegroundColor Gray
Write-Host "  - Project structure tree" -ForegroundColor Gray
Write-Host "`nReady to share!`n" -ForegroundColor Green
