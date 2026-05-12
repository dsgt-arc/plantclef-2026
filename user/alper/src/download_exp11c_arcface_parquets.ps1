# Sequential downloader for everything needed for the kNN-enabled
# scale 3+6 ablation experiment, EXCEPT scale_3x3.parquet itself (which
# is still being extracted on PACE -- run this script again afterwards
# to fetch it once the SBATCH job completes).
#
# Pulls:
#   1. Exp-11C ViT-L LoRA+ArcFace per-tile parquets, scales 4,5,6,7
#      Source: /storage/.../aerten3/exp11c_vitL_lora_arcface/embeddings/test/
#      Each has BOTH teacher_logits_f16 AND emb_arcface_f16 (1024-d)
#
#   2. FAISS index over training-set ArcFace embeddings (ViT-L 1024-d)
#      Source: /storage/.../aerten3/exp11c_vitL_lora_arcface/faiss_index/
#
# Re-runnable: skips files already present locally unless -Force is passed.
# Skipping scale_3x3 is hard-coded -- run again later to pick it up.
#
# Usage:
#   pwsh .\download_exp11c_arcface_parquets.ps1
#   pwsh .\download_exp11c_arcface_parquets.ps1 -Force

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$user = "pace"
$remote_base = "/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/aerten3/exp11c_vitL_lora_arcface"
$dst_base = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads\exp11c_vitL_lora_arcface"

$dst_tiles = "$dst_base\embeddings\test"
$dst_faiss = "$dst_base\faiss_index"
New-Item -Force -ItemType Directory $dst_tiles | Out-Null
New-Item -Force -ItemType Directory $dst_faiss | Out-Null

# Build the work list: (remote path, local path, label, is-scale-3-skip)
$work = @()

# --- 1. Per-tile parquets at scales 4, 5, 6, 7 (skip 3x3) ---
$tiles_remote_dir = "$remote_base/embeddings/test"
Write-Host "Listing per-tile parquets at $tiles_remote_dir ..."
$listing = ssh $user "ls -1 $tiles_remote_dir/*.parquet 2>/dev/null"
$tile_files = $listing | Where-Object { $_ -and $_.Trim() -ne "" }

foreach ($f in $tile_files) {
    $name = Split-Path $f -Leaf
    $work += [PSCustomObject]@{
        Label  = "tile / $name"
        Remote = $f
        Local  = (Join-Path $dst_tiles $name)
    }
}

# --- 2. FAISS index + labels ---
$faiss_remote_dir = "$remote_base/faiss_index"
foreach ($name in @("faiss_index_arcface_geo.bin", "faiss_labels_arcface_geo.npy")) {
    $work += [PSCustomObject]@{
        Label  = "faiss / $name"
        Remote = "$faiss_remote_dir/$name"
        Local  = (Join-Path $dst_faiss $name)
    }
}

$total = $work.Count
if ($total -eq 0) {
    Write-Host "No files to fetch."
    exit 1
}

Write-Host "Total files to fetch: $total"
Write-Host ""

$skipped = 0
$transferred = 0
$failed = 0
$i = 0

foreach ($item in $work) {
    $i++
    if (-not $Force -and (Test-Path $item.Local)) {
        Write-Host "[$i/$total] $($item.Label)  -- already present, skipping"
        $skipped++
        continue
    }

    Write-Host "[$i/$total] $($item.Label)"
    $start = Get-Date
    try {
        scp "${user}:$($item.Remote)" $item.Local
        if ($LASTEXITCODE -ne 0) { throw "scp exit code $LASTEXITCODE" }
        $elapsed = (Get-Date) - $start
        $sizeBytes = (Get-Item $item.Local).Length
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
Write-Host ""
Write-Host "Destination dirs:"
Write-Host "  tile parquets : $dst_tiles"
Write-Host "  FAISS index   : $dst_faiss"
Write-Host ""
Write-Host "Next: when the PACE scale_3x3 extraction finishes, re-run this script;"
Write-Host "      it'll fetch scale_3x3.parquet (remove the skip line in the script,"
Write-Host "      or simply manually scp it the same way)."

if ($failed -gt 0) { exit 1 }
