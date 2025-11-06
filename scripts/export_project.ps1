# Export Project Structure and Code
# This script creates a shareable bundle of your project

param(
    [string]$OutputFile = "export\project_export.zip",
    [switch]$IncludeResults,
    [switch]$IncludeVenv
)

Write-Host "`n=== MIMO GEOMETRY ANALYSIS PROJECT EXPORT ===`n" -ForegroundColor Cyan

# 1. Generate detailed structure
Write-Host "[1/5] Generating project structure..." -ForegroundColor Yellow
tree /F /A > export\project_tree.txt
Write-Host "  ✓ Saved: export\project_tree.txt" -ForegroundColor Green

# 2. Create code archive (exclude venv, results, cache)
Write-Host "`n[2/5] Creating code archive..." -ForegroundColor Yellow

$excludeDirs = @(
    "envs",
    "mimo-geom-dev",
    ".git",
    ".vscode",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "node_modules"
)

if (-not $IncludeResults) {
    $excludeDirs += "results"
}

# Create a list of files to include
$filesToInclude = @()

# Core Python code
$filesToInclude += Get-ChildItem -Path "core" -Recurse -File -Include *.py,*.md,*.txt | 
    Where-Object { $_.DirectoryName -notmatch "__pycache__" }

# Geometry processors
$filesToInclude += Get-ChildItem -Path "geometry_processors" -Recurse -File -Include *.py,*.md |
    Where-Object { $_.DirectoryName -notmatch "__pycache__" }

# Tools and scripts
$filesToInclude += Get-ChildItem -Path "tools" -Recurse -File -Include *.py,*.md
$filesToInclude += Get-ChildItem -Path "scripts" -Recurse -File -Include *.ps1,*.py,*.sh

# Paper materials
$filesToInclude += Get-ChildItem -Path "papers" -Recurse -File -Include *.md,*.tex,*.pdf,*.png

# Root files
$filesToInclude += Get-ChildItem -Path "." -File -Include *.md,*.txt,requirements.txt,.gitignore

Write-Host "  ✓ Found $($filesToInclude.Count) files to export" -ForegroundColor Green

# 3. Create comprehensive README
Write-Host "`n[3/5] Generating export README..." -ForegroundColor Yellow

$readmeContent = @"
# MIMO GEOMETRY ANALYSIS - Project Export

**Export Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Project Version:** ALSS Implementation + SVD Analysis

---

## 📦 Contents

\`\`\`
$((tree /A | Select-Object -First 100) -join "`n")
...
(See project_tree.txt for full structure)
\`\`\`

---

## 🚀 Quick Start

### 1. Setup Environment

\`\`\`powershell
# Create virtual environment
python -m venv mimo-geom-dev

# Activate (Windows)
.\mimo-geom-dev\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Run Benchmarks

\`\`\`powershell
# Basic benchmark
python core\analysis_scripts\run_benchmarks.py \`
  --arrays Z5 --N 7 --algs CoarrayMUSIC \`
  --snr 10 --snapshots 64 --k 2 --trials 10 \`
  --out results\bench\test.csv

# With ALSS enabled
python core\analysis_scripts\run_benchmarks.py \`
  --arrays Z5 --N 7 --algs CoarrayMUSIC \`
  --snr 10 --snapshots 64 --k 2 --trials 10 \`
  --alss on --alss-mode zero --alss-tau 1.0 \`
  --out results\bench\alss_test.csv

# With SVD analysis
python core\analysis_scripts\run_benchmarks.py \`
  --arrays Z5 --N 7 --algs SpatialMUSIC,CoarrayMUSIC \`
  --snr 10 --snapshots 64 --k 2 --trials 10 \`
  --dump-svd --out results\bench\svd_test.csv
\`\`\`

### 3. Generate Plots

\`\`\`powershell
# ALSS performance plots
python tools\plot_alss_updated.py

# SVD analysis
python tools\analyze_svd.py results\svd\
\`\`\`

---

## 📂 Project Structure

### Core Components

- **core/radarpy/algorithms/** - DOA estimation algorithms
  - \`spatial_music.py\` - Physical array MUSIC
  - \`coarray_music.py\` - Virtual array MUSIC
  - \`alss.py\` - Adaptive Lag-Selective Shrinkage
  - \`coarray.py\` - Coarray construction

- **geometry_processors/** - Array geometry analysis
  - \`bases_classes.py\` - Abstract framework
  - \`z4_processor.py\`, \`z5_processor.py\`, \`z6_processor.py\` - Specialized arrays
  - \`ula_processors.py\`, \`nested_processor.py\` - Standard arrays

- **core/analysis_scripts/** - Benchmark runners
  - \`run_benchmarks.py\` - Main benchmark script (supports ALSS, SVD)

- **tools/** - Analysis utilities
  - \`plot_alss_updated.py\` - ALSS performance visualization
  - \`analyze_svd.py\` - SVD/condition number analysis
  - \`merge_sweep_results.py\` - CSV aggregation

- **scripts/** - Automation
  - \`run_sweep_updated.ps1\` - Automated parameter sweeps

### Documentation

- **papers/radarcon2025_alss/** - ALSS paper materials
  - \`ALSS_INTEGRATION_GUIDE.md\` - Integration guide
  - \`ALSS_SWEEP_GUIDE.md\` - Parameter sweep guide
  - \`SVD_ANALYSIS_GUIDE.md\` - SVD analysis guide
  - \`figures/\` - Publication-ready plots (PDF + PNG)

---

## 🔬 Key Features

### 1. ALSS (Adaptive Lag-Selective Shrinkage)
- Reduces estimation variance for virtual covariance matrices
- Configurable modes: 'zero' (toward 0) or 'ar1' (toward AR(1) prior)
- Parameters: tau (strength), coreL (protected lags)

### 2. SVD Analysis
- Captures singular values of Rx (physical) and Rv (virtual) covariance
- Computes condition numbers (κ = σ_max / σ_min)
- Visualizes eigenvalue spectra
- Enables stability analysis

### 3. Array Geometries
- ULA, Nested, Z1-Z6 specialized arrays
- Difference coarray analysis
- Contiguous segment identification
- Performance metrics (L, K_max, Mv)

### 4. Comprehensive Benchmarking
- Multi-parameter sweeps (SNR, snapshots, delta)
- RMSE and resolve rate metrics
- Cramér-Rao Bound (CRB) computation
- CSV output with full metadata

---

## 📊 Results

$(if (Test-Path "results\alss") { 
    "### ALSS Sweep Results`n`n"
    "- **CSV files:** $((Get-ChildItem results\alss\*.csv -Exclude old_backup).Count)`n"
    "- **Total trials:** $((Get-ChildItem results\alss\*.csv -Exclude old_backup).Count * 100)`n"
    "- **Configuration:** SNR: 0,5,10,15 dB; Delta: 10,20,30,45 degrees`n"
} else {
    "No ALSS results included in this export.`n"
})

$(if (Test-Path "results\svd") {
    "### SVD Analysis Results`n`n"
    "- **SVD files:** $((Get-ChildItem results\svd\*.csv -ErrorAction SilentlyContinue).Count)`n"
    "- **Condition number tables:** Available in results/svd/analysis/`n"
} else {
    "No SVD results included in this export.`n"
})

---

## 📖 Documentation Index

1. **ALSS Integration Guide** - \`papers/radarcon2025_alss/ALSS_INTEGRATION_GUIDE.md\`
2. **ALSS Sweep Guide** - \`papers/radarcon2025_alss/ALSS_SWEEP_GUIDE.md\`
3. **SVD Analysis Guide** - \`papers/radarcon2025_alss/SVD_ANALYSIS_GUIDE.md\`
4. **Project Structure** - \`project_tree.txt\`

---

## 🛠️ Dependencies

\`\`\`
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0
\`\`\`

See \`requirements.txt\` for complete list.

---

## 📝 Citation

If you use this code in your research, please cite:

\`\`\`
@inproceedings{your_paper_2025,
  title={Adaptive Lag-Selective Shrinkage for MIMO Coarray DOA Estimation},
  author={Your Name},
  booktitle={IEEE RadarCon 2025},
  year={2025}
}
\`\`\`

---

## 📧 Contact

For questions or issues, please contact: [Your Email]

---

**Generated:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@

$readmeContent | Out-File -FilePath "export\README.md" -Encoding UTF8
Write-Host "  ✓ Saved: export\README.md" -ForegroundColor Green

# 4. Copy key files to export directory
Write-Host "`n[4/5] Copying files to export..." -ForegroundColor Yellow

$exportStructure = @{
    "core" = "export\core"
    "geometry_processors" = "export\geometry_processors"
    "tools" = "export\tools"
    "scripts" = "export\scripts"
    "papers" = "export\papers"
}

foreach ($source in $exportStructure.Keys) {
    $dest = $exportStructure[$source]
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Recurse -Force -Exclude __pycache__,*.pyc
        Write-Host "  ✓ Copied: $source -> $dest" -ForegroundColor Green
    }
}

# Copy root files
Copy-Item -Path "requirements.txt" -Destination "export\" -Force -ErrorAction SilentlyContinue
Copy-Item -Path "README.md" -Destination "export\README_ORIGINAL.md" -Force -ErrorAction SilentlyContinue
Copy-Item -Path ".gitignore" -Destination "export\" -Force -ErrorAction SilentlyContinue

# 5. Create ZIP archive
Write-Host "`n[5/5] Creating ZIP archive..." -ForegroundColor Yellow

if (Test-Path $OutputFile) {
    Remove-Item $OutputFile -Force
}

Compress-Archive -Path "export\*" -DestinationPath $OutputFile -Force
$zipSize = (Get-Item $OutputFile).Length / 1MB

Write-Host "  ✓ Created: $OutputFile ($([math]::Round($zipSize, 2)) MB)" -ForegroundColor Green

# Summary
Write-Host "`n=== EXPORT COMPLETE ===`n" -ForegroundColor Cyan
Write-Host "📦 Archive: $OutputFile" -ForegroundColor White
Write-Host "📄 Files: $($filesToInclude.Count)" -ForegroundColor White
Write-Host "💾 Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor White
Write-Host "`nContents:" -ForegroundColor Cyan
Write-Host "  • Project structure (project_tree.txt)" -ForegroundColor White
Write-Host "  • All Python source code" -ForegroundColor White
Write-Host "  • Documentation (Markdown files)" -ForegroundColor White
Write-Host "  • Analysis scripts and tools" -ForegroundColor White
Write-Host "  • Paper materials (figures, LaTeX)" -ForegroundColor White
if ($IncludeResults) {
    Write-Host "  • Benchmark results" -ForegroundColor White
}
Write-Host "`nTo share: Send the file $OutputFile`n" -ForegroundColor Yellow
