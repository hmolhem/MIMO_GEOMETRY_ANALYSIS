#!/usr/bin/env pwsh
# ALSS SNR Sweep Script
# Sweeps SNR values for multiple angular separations to generate comprehensive ablation data

param(
    [int[]]$SNRs = @(-5, 0, 5, 10, 15),
    [int[]]$Deltas = @(2, 13),
    [int]$Trials = 100,
    [int]$Snapshots = 64,
    [string]$OutputDir = "results\alss"
)

Write-Host "`n╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              ALSS SNR SWEEP - COMPREHENSIVE ABLATION              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "📊 SWEEP PARAMETERS:`n" -ForegroundColor Green
Write-Host "   SNR values:    $($SNRs -join ', ') dB" -ForegroundColor White
Write-Host "   Delta values:  $($Deltas -join ', ')°" -ForegroundColor White
Write-Host "   Snapshots:     $Snapshots" -ForegroundColor White
Write-Host "   Trials/point:  $Trials" -ForegroundColor White
Write-Host "   Output dir:    $OutputDir`n" -ForegroundColor White

$TotalRuns = $SNRs.Count * $Deltas.Count * 2  # x2 for baseline + ALSS
Write-Host "⏱️  Total runs: $TotalRuns (baseline + ALSS for each SNR/delta combo)" -ForegroundColor Yellow
Write-Host "`n" -ForegroundColor White

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$RunCount = 0
$StartTime = Get-Date

foreach ($delta in $Deltas) {
    Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "   DELTA = ${delta}° (sources at ±$($delta/2)°)" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════════`n" -ForegroundColor Cyan
    
    foreach ($snr in $SNRs) {
        $RunCount++
        $Progress = [math]::Round(($RunCount / $TotalRuns) * 100, 1)
        
        Write-Host "[$RunCount/$TotalRuns - ${Progress}%] " -ForegroundColor Yellow -NoNewline
        Write-Host "SNR=${snr}dB, Δ=${delta}°" -ForegroundColor White
        
        # Baseline (ALSS OFF)
        Write-Host "  → Running BASELINE... " -ForegroundColor Gray -NoNewline
        $BaselineOut = "$OutputDir\baseline_Z5_M${Snapshots}_d${delta}_snr${snr}.csv"
        
        python core\analysis_scripts\run_benchmarks.py `
            --arrays Z5 --N 7 --algs CoarrayMUSIC `
            --lambda_factor 2.0 --snr $snr --snapshots $Snapshots `
            --k 2 --delta $delta --trials $Trials `
            --alss off `
            --out $BaselineOut 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓" -ForegroundColor Green
        } else {
            Write-Host "✗ FAILED" -ForegroundColor Red
        }
        
        $RunCount++
        $Progress = [math]::Round(($RunCount / $TotalRuns) * 100, 1)
        
        Write-Host "[$RunCount/$TotalRuns - ${Progress}%] " -ForegroundColor Yellow -NoNewline
        Write-Host "SNR=${snr}dB, Δ=${delta}°" -ForegroundColor White
        
        # ALSS ON
        Write-Host "  → Running ALSS...     " -ForegroundColor Gray -NoNewline
        $ALSSOut = "$OutputDir\alss_Z5_M${Snapshots}_d${delta}_snr${snr}.csv"
        
        python core\analysis_scripts\run_benchmarks.py `
            --arrays Z5 --N 7 --algs CoarrayMUSIC `
            --lambda_factor 2.0 --snr $snr --snapshots $Snapshots `
            --k 2 --delta $delta --trials $Trials `
            --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
            --out $ALSSOut 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓" -ForegroundColor Green
        } else {
            Write-Host "✗ FAILED" -ForegroundColor Red
        }
    }
}

Write-Host "`n═══════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "   MERGING ALL RESULTS INTO SINGLE CSV" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════`n" -ForegroundColor Green

# Merge all CSVs
$AllCSVs = Get-ChildItem "$OutputDir\*.csv" -Exclude "all_runs.csv"
Write-Host "Found $($AllCSVs.Count) CSV files to merge..." -ForegroundColor White

$MergedData = $AllCSVs | ForEach-Object { Import-Csv $_.FullName }
$MergedData | Export-Csv "$OutputDir\all_runs.csv" -NoTypeInformation

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "`n╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    ✓ SWEEP COMPLETE                                ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Host "📊 RESULTS:" -ForegroundColor Cyan
Write-Host "   Individual files: $OutputDir\baseline_*.csv, alss_*.csv" -ForegroundColor White
Write-Host "   Merged file:      $OutputDir\all_runs.csv" -ForegroundColor White
Write-Host "   Total trials:     $(($MergedData | Measure-Object).Count)" -ForegroundColor White
Write-Host "   Duration:         $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor White

Write-Host "`n💡 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "   1. Analyze results: python tools\analyze_alss_results.py $OutputDir\all_runs.csv" -ForegroundColor White
Write-Host "   2. Generate plots:  python tools\plot_alss_figures.py (update paths first)" -ForegroundColor White
Write-Host "   3. Review data:     code $OutputDir\all_runs.csv`n" -ForegroundColor White
