# Activate Virtual Environment - mimo-geom-dev
# Usage: .\activate_venv.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Activating mimo-geom-dev Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$venvPath = Join-Path $PSScriptRoot "envs\mimo-geom-dev\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    Write-Host "Found virtual environment at: $venvPath" -ForegroundColor Green
    Write-Host "Attempting activation..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        # Try to activate with & operator
        & $venvPath
        Write-Host ""
        Write-Host "✓ Virtual environment activated successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Python version:" -ForegroundColor Cyan
        python --version
        Write-Host ""
        Write-Host "Python location:" -ForegroundColor Cyan
        where.exe python | Select-Object -First 1
        Write-Host ""
        Write-Host "Installed packages:" -ForegroundColor Cyan
        pip list | Select-String "numpy|pandas|matplotlib"
        Write-Host ""
        Write-Host "You can now run benchmarks and tests!" -ForegroundColor Green
        Write-Host ""
    }
    catch {
        Write-Host ""
        Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
        Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        Write-Host "SOLUTION: Run this command first:" -ForegroundColor Yellow
        Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
        Write-Host ""
        Write-Host "Then try again: .\activate_venv.ps1" -ForegroundColor White
        Write-Host ""
    }
}
else {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Expected location: $venvPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please check that the virtual environment exists." -ForegroundColor Yellow
    Write-Host ""
}
