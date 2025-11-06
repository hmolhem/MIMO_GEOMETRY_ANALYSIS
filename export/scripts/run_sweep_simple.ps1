# Simple SNR Sweep - Run all combinations
# SNR: -5, 0, 5, 10, 15 dB
# Delta: 2, 13 degrees
# Total: 20 runs (5 SNR x 2 delta x 2 modes)

Write-Host "`nStarting ALSS SNR Sweep..." -ForegroundColor Cyan
Write-Host "SNR: -5, 0, 5, 10, 15 dB" -ForegroundColor White
Write-Host "Delta: 2, 13 degrees" -ForegroundColor White
Write-Host "Total runs: 20 (this will take ~30-40 minutes)`n" -ForegroundColor Yellow

$count = 0
$total = 20

# Delta = 2 degrees
Write-Host "=== DELTA = 2 degrees ===" -ForegroundColor Cyan

foreach ($snr in @(-5, 0, 5, 10, 15)) {
    $count++
    Write-Host "[$count/$total] Baseline: SNR=${snr}dB, Delta=2..." -ForegroundColor Yellow
    python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --algs CoarrayMUSIC --lambda_factor 2.0 --snr $snr --snapshots 64 --k 2 --delta 2 --trials 100 --alss off --out results\alss\baseline_Z5_M64_d2_snr${snr}.csv
    
    $count++
    Write-Host "[$count/$total] ALSS: SNR=${snr}dB, Delta=2..." -ForegroundColor Yellow
    python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --algs CoarrayMUSIC --lambda_factor 2.0 --snr $snr --snapshots 64 --k 2 --delta 2 --trials 100 --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 --out results\alss\alss_Z5_M64_d2_snr${snr}.csv
}

# Delta = 13 degrees
Write-Host "`n=== DELTA = 13 degrees ===" -ForegroundColor Cyan

foreach ($snr in @(-5, 0, 5, 10, 15)) {
    $count++
    Write-Host "[$count/$total] Baseline: SNR=${snr}dB, Delta=13..." -ForegroundColor Yellow
    python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --algs CoarrayMUSIC --lambda_factor 2.0 --snr $snr --snapshots 64 --k 2 --delta 13 --trials 100 --alss off --out results\alss\baseline_Z5_M64_d13_snr${snr}.csv
    
    $count++
    Write-Host "[$count/$total] ALSS: SNR=${snr}dB, Delta=13..." -ForegroundColor Yellow
    python core\analysis_scripts\run_benchmarks.py --arrays Z5 --N 7 --algs CoarrayMUSIC --lambda_factor 2.0 --snr $snr --snapshots 64 --k 2 --delta 13 --trials 100 --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 --out results\alss\alss_Z5_M64_d13_snr${snr}.csv
}

# Merge all results
Write-Host "`nMerging all CSVs..." -ForegroundColor Cyan
Get-ChildItem results\alss\*.csv -Exclude "all_runs.csv" | ForEach-Object { Import-Csv $_.FullName } | Export-Csv results\alss\all_runs.csv -NoTypeInformation

Write-Host "`n" -NoNewline
Write-Host "Sweep complete! Results in: results\alss\all_runs.csv" -ForegroundColor Green
Write-Host "Run: python tools\plot_alss_sweep.py results\alss\all_runs.csv" -ForegroundColor Yellow
