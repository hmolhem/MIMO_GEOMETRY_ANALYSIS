# Updated ALSS SNR/Delta Sweep
# SNR: 0, 5, 10, 15 dB
# Delta: 10, 20, 30, 45 degrees
# Total: 32 runs (4 SNR × 4 Delta × 2 modes)

$SNRs = @(0, 5, 10, 15)
$Deltas = @(10, 20, 30, 45)
$Trials = 100
$Snapshots = 64

Write-Host "`nStarting Updated ALSS SNR/Delta Sweep..." -ForegroundColor Green
Write-Host "SNR: $($SNRs -join ', ') dB" -ForegroundColor Cyan
Write-Host "Delta: $($Deltas -join ', ') degrees" -ForegroundColor Cyan
Write-Host "Total runs: 32 (this will take ~60-70 minutes)`n" -ForegroundColor Yellow

$runCount = 0
$totalRuns = $SNRs.Count * $Deltas.Count * 2

foreach ($delta in $Deltas) {
    Write-Host "`n=== DELTA = $delta degrees ===" -ForegroundColor Magenta
    
    foreach ($snr in $SNRs) {
        # Baseline run
        $runCount++
        Write-Host "[$runCount/$totalRuns] Baseline: SNR=${snr}dB, Delta=$delta..." -ForegroundColor White
        
        $outFile = "results\alss\baseline_Z5_M64_d${delta}_snr${snr}.csv"
        
        if (Test-Path $outFile) {
            Write-Host "  -> Skipping (file exists)" -ForegroundColor Gray
        } else {
            python core\analysis_scripts\run_benchmarks.py `
                --arrays Z5 --N 7 --algs CoarrayMUSIC `
                --lambda_factor 2.0 --snr $snr --snapshots $Snapshots `
                --k 2 --delta $delta --trials $Trials `
                --alss off `
                --out $outFile
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[Saved] Bench CSV -> $outFile" -ForegroundColor Green
            } else {
                Write-Host "[ERROR] Failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit 1
            }
        }
        
        # ALSS run
        $runCount++
        Write-Host "[$runCount/$totalRuns] ALSS: SNR=${snr}dB, Delta=$delta..." -ForegroundColor White
        
        $outFile = "results\alss\alss_Z5_M64_d${delta}_snr${snr}.csv"
        
        if (Test-Path $outFile) {
            Write-Host "  -> Skipping (file exists)" -ForegroundColor Gray
        } else {
            python core\analysis_scripts\run_benchmarks.py `
                --arrays Z5 --N 7 --algs CoarrayMUSIC `
                --lambda_factor 2.0 --snr $snr --snapshots $Snapshots `
                --k 2 --delta $delta --trials $Trials `
                --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
                --out $outFile
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[Saved] ALSS CSV -> $outFile" -ForegroundColor Green
            } else {
                Write-Host "[ERROR] Failed with exit code $LASTEXITCODE" -ForegroundColor Red
                exit 1
            }
        }
    }
}

Write-Host "`n=== Sweep Complete! ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Merge CSVs: python tools\merge_sweep_results.py" -ForegroundColor White
Write-Host "  2. Plot results: python tools\plot_alss_sweep.py results\alss\all_runs.csv`n" -ForegroundColor White
