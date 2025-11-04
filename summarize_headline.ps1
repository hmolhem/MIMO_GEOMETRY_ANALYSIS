# PowerShell script to summarize headline benchmark with correct CRB interpretation
# Usage: .\summarize_headline.ps1 [path_to_csv]

param(
    [string]$CsvPath = "results/bench/headline.csv",
    [string]$CrbPath = "results/bench/crb_overlay.csv"
)

if (-not (Test-Path $CsvPath)) {
    Write-Host "[ERROR] CSV not found: $CsvPath" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Headline Benchmark Summary ===" -ForegroundColor Cyan
Write-Host "Data: $CsvPath`n"

# Load main results
$data = Import-Csv $CsvPath

# Load CRB if available
$crb = @{}
if (Test-Path $CrbPath) {
    Import-Csv $CrbPath | ForEach-Object {
        $key = "$($_.array)|$($_.alg)|$($_.SNR_dB)|$($_.snapshots)|$($_.delta_deg)"
        $crb[$key] = [double]$_.crb_deg
    }
    Write-Host "[CRB overlay loaded: $($crb.Count) entries]`n" -ForegroundColor Green
}

# Group by configuration
$summary = $data | Group-Object array, alg, SNR_dB, snapshots, delta_deg | ForEach-Object {
    $g = $_.Group
    $arr = $g[0].array
    $alg = $g[0].alg
    $snr = $g[0].SNR_dB
    $M = $g[0].snapshots
    $delta = $g[0].delta_deg
    
    $avgRmse = ($g.rmse_deg | Measure-Object -Average).Average
    $resolve = ($g.resolved | Measure-Object -Sum).Sum / $g.Count * 100
    $Mv = $g[0].Mv
    
    # Lookup CRB
    $key = "$arr|$alg|$snr|$M|$delta"
    $crbVal = $crb[$key]
    
    if ($crbVal -and $crbVal -gt 0) {
        $ratio = $avgRmse / $crbVal
        $ratioStr = "{0:F2}× above CRB" -f $ratio
    } else {
        $ratioStr = "CRB N/A"
    }
    
    [pscustomobject]@{
        Array = $arr
        Algorithm = $alg
        SNR_dB = $snr
        Snapshots = $M
        Delta_deg = $delta
        AvgRMSE = [math]::Round($avgRmse, 4)
        Resolve_pct = [math]::Round($resolve, 1)
        Mv = $Mv
        CRB_deg = if ($crbVal) { [math]::Round($crbVal, 5) } else { "N/A" }
        Ratio = $ratioStr
    }
}

# Display grouped by array/algorithm
$summary | Sort-Object Array, Algorithm, SNR_dB, Snapshots, Delta_deg | Format-Table -AutoSize

# Headline point (SNR=10, M=256, Δ=2)
Write-Host "`n=== Headline Point (SNR=10dB, M=256, Δθ=2°) ===" -ForegroundColor Yellow
$headline = $summary | Where-Object { $_.SNR_dB -eq 10 -and $_.Snapshots -eq 256 -and $_.Delta_deg -eq 2 }
$headline | Format-Table -AutoSize

Write-Host "`nInterpretation: RMSE/CRB ratio shows how many times above the theoretical bound the estimator performs." -ForegroundColor Cyan
Write-Host "Lower ratios indicate better performance (closer to CRB). Ratios > 1 are expected and correct.`n"
