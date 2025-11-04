# Quick smoke test for headline benchmark (reduced trials for fast validation)
# Run this first to verify everything works before the full 100-trial sweep

.\mimo-geom-dev\Scripts\python.exe -m analysis_scripts.run_benchmarks `
  --arrays Z4,Z5,ULA `
  --algs SpatialMUSIC,CoarrayMUSIC `
  --N 7 --d 1.0 --lambda_factor 2.0 `
  --snr 0,10 `
  --snapshots 64,256 `
  --k 2 --delta 2 `
  --trials 10 `
  --out results/bench/headline_smoke.csv --save-crb

Write-Host "`n[Smoke Test Complete] Verify results:" -ForegroundColor Green
$data = Import-Csv results/bench/headline_smoke.csv
$summary = $data | Group-Object array, alg | ForEach-Object {
    $g = $_.Group
    [pscustomobject]@{
        Config = $_.Name
        AvgRMSE = [math]::Round((($g.rmse_deg | Measure-Object -Average).Average), 3)
        Resolve = [math]::Round((($g.resolved | Measure-Object -Sum).Sum / $g.Count * 100), 1)
        Mv = $g[0].Mv
    }
}
$summary | Format-Table -AutoSize
