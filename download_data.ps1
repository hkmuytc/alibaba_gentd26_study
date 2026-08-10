# Downloads the GenTD26 dataset (Alibaba cluster-trace-v2026-GenAI) into data/raw/
# Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File download_data.ps1

$ErrorActionPreference = "Stop"

$baseUrl = "https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2026-GenAI"
$rawDir  = Join-Path $PSScriptRoot "data" "raw"

if (-not (Test-Path $rawDir)) {
    New-Item -ItemType Directory -Path $rawDir -Force | Out-Null
}

$files = @(
    "pod_gpu_duty_cycle_anon.tar.gz",
    "pod_gpu_memory_used_bytes_anon.tar.gz",
    "pod_memory_util_anon.tar.gz",
    "qps.tar.gz",
    "data_trace_processed.tar.gz"
)

Write-Host "Downloading GenTD26 dataset to $rawDir ..."

foreach ($file in $files) {
    $csv      = $file -replace '\.tar\.gz$', '.csv'
    $csvPath  = Join-Path $rawDir $csv
    $tgzPath  = Join-Path $rawDir $file

    if (Test-Path $csvPath) {
        Write-Host "  [skip] $csv already exists"
        continue
    }

    Write-Host "  Downloading $file ..."
    Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile $tgzPath -UseBasicParsing

    Write-Host "  Extracting $file ..."
    tar -xzf $tgzPath -C $rawDir

    Remove-Item $tgzPath
}

Write-Host "`nDone. Files in $rawDir :"
Get-ChildItem "$rawDir\*.csv" | ForEach-Object {
    Write-Host ("  {0,-45} {1:N1} MB" -f $_.Name, ($_.Length / 1MB))
}
