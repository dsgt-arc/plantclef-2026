# Sequential downloader: pulls every ViT-B *.parquet (classifier-only, no ArcFace)
# down to the local destination via plain scp, one file at a time.
#
# Source layout on PACE (Exp-7 multiscale extraction):
#   /storage/.../parquet/plantclef_2026_test/exp7_multiscale/scale_{3,5,6,7}x{3,5,6,7}.parquet
#   /storage/.../parquet/plantclef_2026_test/phase1_tiles_test_jpeg85/*.parquet   (scale 4 — separate path)
#
# These contain teacher_logits_f16 (classifier head) only. They are exactly
# what's needed for the Abl-3 "ViT-B vs ViT-L without kNN ensembling" ablation.
# The richer Exp-9B variant (with ArcFace embeddings + FAISS for kNN) is NOT
# pulled here — drop kNN, keep only the bare backbone outputs.
#
# Re-running the script is safe — it skips files already present locally.
#
# Usage:
#   pwsh .\download_vitB_multiscale_parquets.ps1
#   pwsh .\download_vitB_multiscale_parquets.ps1 -Force

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$user = "pace"   # ~/.ssh/config alias (HostName=login-phoenix.pace.gatech.edu, User=aerten3)

# Two remote source directories. The local destination preserves subdirs to avoid name collisions.
$remoteRoot = "/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/aerten3/parquet/plantclef_2026_test"
$sources = @(
    @{ Subdir = "exp7_multiscale";          Remote = "$remoteRoot/exp7_multiscale" },
    @{ Subdir = "phase1_tiles_test_jpeg85"; Remote = "$remoteRoot/phase1_tiles_test_jpeg85" }
)

$dstRoot = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads\vitB_multiscale"
New-Item -Force -ItemType Directory $dstRoot | Out-Null

# Collect a flat work list across both source dirs
Write-Host "Listing ViT-B parquet files on PACE..."
$work = @()
foreach ($s in $sources) {
    $listing = ssh $user "ls -1 $($s.Remote)/*.parquet 2>/dev/null"
    $files = $listing | Where-Object { $_ -and $_.Trim() -ne "" }
    foreach ($f in $files) {
        $work += [PSCustomObject]@{
            Remote = $f
            Subdir = $s.Subdir
            Name   = Split-Path $f -Leaf
        }
    }
}

if (-not $work) {
    Write-Host "No parquet files found in either remote directory."
    exit 1
}

$total = $work.Count
Write-Host "Found $total parquet file(s) across $($sources.Count) source dir(s). Destination: $dstRoot"
Write-Host ""

$skipped = 0
$transferred = 0
$failed = 0
$i = 0

foreach ($item in $work) {
    $i++
    $localDir = Join-Path $dstRoot $item.Subdir
    New-Item -Force -ItemType Directory $localDir | Out-Null
    $localPath = Join-Path $localDir $item.Name
    $tag = "$($item.Subdir)/$($item.Name)"

    if (-not $Force -and (Test-Path $localPath)) {
        Write-Host "[$i/$total] $tag  -- already present, skipping"
        $skipped++
        continue
    }

    Write-Host "[$i/$total] $tag"
    $start = Get-Date
    try {
        scp "${user}:$($item.Remote)" $localPath
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
Write-Host "  destination : $dstRoot"

if ($failed -gt 0) { exit 1 }
