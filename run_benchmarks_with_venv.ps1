# Run Benchmarks with Virtual Environment
# This script ensures the virtual environment is activated before running benchmarks
# Usage: .\run_benchmarks_with_venv.ps1 [benchmark_args]

param(
    [string]$Arrays = "Z5",
    [int]$N = 7,
    [int]$Trials = 100,
    [switch]$WithCoupling,
    [string]$AdditionalArgs = ""
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MIMO Array Benchmark Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv is already activated
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
$venvPython = Join-Path $PSScriptRoot "envs\mimo-geom-dev\Scripts\python.exe"

if ($pythonPath -eq $venvPython) {
    Write-Host "✓ Virtual environment already active" -ForegroundColor Green
} else {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    $activateScript = Join-Path $PSScriptRoot "envs\mimo-geom-dev\Scripts\Activate.ps1"
    
    if (Test-Path $activateScript) {
        try {
            & $activateScript
            Write-Host "✓ Virtual environment activated" -ForegroundColor Green
        } catch {
            Write-Host "WARNING: Could not activate via PowerShell" -ForegroundColor Yellow
            Write-Host "Attempting to use venv Python directly..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "ERROR: Virtual environment not found at: $activateScript" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Python Environment Info:" -ForegroundColor Cyan
Write-Host "  Python: $(python --version)" -ForegroundColor White
Write-Host "  Location: $(python -c 'import sys; print(sys.executable)')" -ForegroundColor White
Write-Host ""

# Build benchmark command
$benchmarkScript = "core\analysis_scripts\run_benchmarks.py"
$command = "python $benchmarkScript --arrays $Arrays --N $N --trials $Trials"

if ($WithCoupling) {
    $command += " --coupling exponential --coupling-strength 0.3"
    Write-Host "Note: Running WITH mutual coupling enabled" -ForegroundColor Yellow
}

if ($AdditionalArgs) {
    $command += " $AdditionalArgs"
}

Write-Host "Running benchmark:" -ForegroundColor Cyan
Write-Host "  $command" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run the benchmark
Invoke-Expression $command

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Benchmark Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
