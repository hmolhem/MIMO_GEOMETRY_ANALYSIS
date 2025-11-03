# analysis_scripts/run_z4_z5_sweep.ps1
# PowerShell script to run Z4 and Z5 demos across multiple N values for paper comparison

param(
    [int[]]$NSizes = @(7, 9, 11, 13, 15),
    [double]$d = 1.0,
    [switch]$LaTeX,
    [switch]$SaveArtifacts,
    [switch]$Quiet
)

Write-Host "=== Z4/Z5 N-Sweep Analysis for Paper ===" -ForegroundColor Green
Write-Host "N values: $($NSizes -join ', ')" -ForegroundColor Cyan  
Write-Host "Physical spacing (d): $d" -ForegroundColor Cyan
Write-Host "Save artifacts: $SaveArtifacts" -ForegroundColor Cyan
Write-Host "LaTeX output: $LaTeX" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the right directory
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Create results directories
if ($SaveArtifacts) {
    New-Item -ItemType Directory -Force -Path "results/summaries" | Out-Null
    New-Item -ItemType Directory -Force -Path "results/geometries" | Out-Null
    New-Item -ItemType Directory -Force -Path "results/coarrays" | Out-Null
}

foreach ($N in $NSizes) {
    if (-not $Quiet) {
        Write-Host "--- Processing N=$N ---" -ForegroundColor Yellow
    }
    
    # Run Z4 demo
    if (-not $Quiet) {
        Write-Host "  Running Z4..." -ForegroundColor Blue
    }
    
    $z4Args = @("--N", $N, "--d", $d, "--assert")
    if ($SaveArtifacts) {
        $z4Args += @("--save-csv", "--save-json")
    }
    if ($LaTeX) {
        $z4Args += "--latex"
    }
    
    try {
        $z4Output = & python "analysis_scripts/run_z4_demo.py" @z4Args 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Z4 demo returned exit code $LASTEXITCODE for N=$N"
        }
    } catch {
        Write-Error "Z4 demo failed for N=$N`: $_"
        continue
    }
    
    # Run Z5 demo
    if (-not $Quiet) {
        Write-Host "  Running Z5..." -ForegroundColor Blue
    }
    
    $z5Args = @("--N", $N, "--d", $d, "--assert")
    if ($SaveArtifacts) {
        $z5Args += @("--save-csv", "--save-json")
    }
    if ($LaTeX) {
        $z5Args += "--latex"
    }
    
    try {
        $z5Output = & python "analysis_scripts/run_z5_demo.py" @z5Args 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Z5 demo returned exit code $LASTEXITCODE for N=$N"
        }
    } catch {
        Write-Error "Z5 demo failed for N=$N`: $_"
        continue
    }
    
    # If LaTeX mode, extract and display the LaTeX rows
    if ($LaTeX) {
        Write-Host "LaTeX rows for N=$N`:" -ForegroundColor Magenta
        
        # Extract LaTeX line from Z4 output
        $z4LaTeX = ($z4Output | Where-Object { $_ -match "^$N\s*&" }) -join ""
        if ($z4LaTeX) {
            Write-Host "Z4: $z4LaTeX" -ForegroundColor White
        } else {
            Write-Host "Z4: (No LaTeX output found)" -ForegroundColor Gray
        }
        
        # Extract LaTeX line from Z5 output  
        $z5LaTeX = ($z5Output | Where-Object { $_ -match "^$N\s*&" }) -join ""
        if ($z5LaTeX) {
            Write-Host "Z5: $z5LaTeX" -ForegroundColor White
        } else {
            Write-Host "Z5: (No LaTeX output found)" -ForegroundColor Gray
        }
        Write-Host ""
    }
    
    if (-not $Quiet) {
        Write-Host "  Complete: N=$N" -ForegroundColor Green
    }
}

Write-Host "=== N-Sweep Complete ===" -ForegroundColor Green

if ($SaveArtifacts) {
    Write-Host ""
    Write-Host "Generated artifacts:" -ForegroundColor Cyan
    Write-Host "  Summaries: results/summaries/z{4,5}_summary_N*_d$d.csv" -ForegroundColor White
    Write-Host "  JSON: results/summaries/z{4,5}_run_N*_d$d.json" -ForegroundColor White
    Write-Host "  Geometries: results/geometries/z{4,5}_N*_d$d_sensors.csv" -ForegroundColor White
    Write-Host "  Coarrays: results/coarrays/z{4,5}_N*_d$d_lags.txt" -ForegroundColor White
}

if ($LaTeX) {
    Write-Host ""
    Write-Host "Tip: Copy LaTeX rows above into your paper table." -ForegroundColor Yellow
}