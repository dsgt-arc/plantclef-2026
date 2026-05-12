# Sequential downloader: pulls every *.parquet from
#   /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/aerten3/parquet/exp13_vit_dinov2_large_phase2v3/
# down to the local destination via plain scp, one file at a time.
#
# Re-running the script is safe — it skips files already present locally.
#
# Usage:
#   pwsh .\download_phase2v3_parquets.ps1
#   pwsh .\download_phase2v3_parquets.ps1 -Force      # re-download everything

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$user = "pace"   # uses Host alias from ~/.ssh/config (HostName=login-phoenix.pace.gatech.edu, User=aerten3)
$remote = "/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/aerten3/parquet/exp13_vit_dinov2_large_phase2v3"
$dst = "C:\Users\alper\Documents\PlantCLEF_2026_Local\pace_parquets\exp13_vit_dinov2_large_phase2v3"

New-Item -Force -ItemType Directory $dst | Out-Null

Write-Host "Listing parquet files on PACE..."
$listing = ssh $user "ls -1 $remote/*.parquet 2>/dev/null"
$files = $listing | Where-Object { $_ -and $_.Trim() -ne "" }

if (-not $files) {
    Write-Host "No parquet files found at $remote"
    exit 1
}

$total = @($files).Count
Write-Host "Found $total parquet file(s). Destination: $dst"
Write-Host ""

$skipped = 0
$transferred = 0
$failed = 0
$i = 0

foreach ($f in $files) {
    $i++
    $name = Split-Path $f -Leaf
    $localPath = Join-Path $dst $name

    if (-not $Force -and (Test-Path $localPath)) {
        Write-Host "[$i/$total] $name  -- already present, skipping"
        $skipped++
        continue
    }

    Write-Host "[$i/$total] $name"
    $start = Get-Date
    try {
        scp "${user}:$f" $localPath
        if ($LASTEXITCODE -ne 0) { throw "scp exit code $LASTEXITCODE" }
        $elapsed = (Get-Date) - $start
        $sizeBytes = (Get-Item $localPath).Length
        $sizeMb = [math]::Round($sizeBytes / 1MB, 1)
        $rateMbps = if ($elapsed.TotalSeconds -gt 0) {
            [math]::Round($sizeMb / $elapsed.TotalSeconds, 1)
        } else { 0 }
        Write-Host "         ok  ($sizeMb MB in $([math]::Round($elapsed.TotalSeconds,1)) s, $rateMbps MB/s)"
        $transferred++
    } catch {
        Write-Host "         FAILED: $($_.Exception.Message)" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "Summary:"
Write-Host "  transferred : $transferred"
Write-Host "  skipped     : $skipped"
Write-Host "  failed      : $failed"
Write-Host "  total       : $total"
Write-Host "  destination : $dst"

if ($failed -gt 0) { exit 1 }
