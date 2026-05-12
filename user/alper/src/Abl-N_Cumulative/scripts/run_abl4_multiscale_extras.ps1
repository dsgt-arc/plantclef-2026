# Extra multi-scale variants on top of the main Abl-4 sweep:
#   7     -- scale 7x7 only            ( 49 tiles/image)
#   8     -- scale 8x8 only            ( 64 tiles/image)
#   45    -- scales 4+5                ( 41 tiles/image)
#   3456  -- scales 3+4+5+6            ( 86 tiles/image)
#
# Run AFTER (or in parallel with, if you have RAM headroom) the main
# run_abl4_multiscale.ps1 sweep.
#
# All other settings identical to the main sweep (Abl-3 cumulative config:
# no kNN, no temporal fusion, no habitat-fit demotion, no masks, no admission).

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABL2  = "$REPO\user\alper\src\Abl-2_KNN"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"
$SCALE_DIR = "$PACE_DL\exp13_vit_dinov2_large_phase2v3"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"
$TOPK = "10"
$HAB_BETA = "1.0"
$HAB_R = "250"
$HAB_N = "40"

$OUT_DIR  = "$ABLN\outputs\abl4_multiscale"
$PREF_DIR = "$OUT_DIR\prefusion"
New-Item -Force -ItemType Directory $OUT_DIR  | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR | Out-Null

function Scales([int[]]$scales) {
    return $scales | ForEach-Object { "$SCALE_DIR\scale_${_}x${_}.parquet" }
}

$variants = @(
    @{ tag = "7";    scales = @(7) },
    @{ tag = "8";    scales = @(8) },
    @{ tag = "45";   scales = @(4,5) },
    @{ tag = "3456"; scales = @(3,4,5,6) }
)

$all_scales_needed = ($variants | ForEach-Object { $_.scales } | Sort-Object -Unique)
foreach ($s in $all_scales_needed) {
    $p = "$SCALE_DIR\scale_${s}x${s}.parquet"
    if (-not (Test-Path $p)) {
        Write-Host "ERROR: missing local scale parquet: $p" -ForegroundColor Red
        exit 1
    }
}

foreach ($v in $variants) {
    $tag = "abl4_scale$($v.tag)"
    Write-Host "==========================================================="
    Write-Host "VARIANT: $tag  -- scales: $($v.scales -join ',')"
    Write-Host "==========================================================="

    $scale_paths = Scales $v.scales

    & $PYTHON "$ABL2\scripts\make_vitL_prefusion_no_knn.py" `
        --vit-scales $scale_paths `
        --class-mask-npy $GEO_MASK `
        --out-dir $PREF_DIR `
        --temperature 1.5 `
        --image-topk 64 `
        --tag $tag
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at $tag prefusion" -ForegroundColor Red; exit 1 }

    $PREFUSION = "$PREF_DIR\${tag}_prefusion.parquet"
    if (-not (Test-Path $PREFUSION)) {
        Write-Host "ERROR: prefusion missing at $PREFUSION" -ForegroundColor Red
        exit 1
    }

    & $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
        --vit-prefusion-parquet $PREFUSION `
        --species-map $SPECIES_MAP `
        --class-mask-npy $GEO_MASK `
        --out-dir $OUT_DIR `
        --thresholds $THR `
        --topk $TOPK `
        --vit-aggregator max `
        --tag $tag
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at $tag ensemble" -ForegroundColor Red; exit 1 }

    $ENS_PARQUET = "$OUT_DIR\${tag}_w1.00_ensemble.parquet"
    if (-not (Test-Path $ENS_PARQUET)) {
        Write-Host "ERROR: ensemble parquet missing at $ENS_PARQUET" -ForegroundColor Red
        exit 1
    }

    $OUT_CSV = "$OUT_DIR\${tag}.csv"
    & $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
        --ensemble-parquet $ENS_PARQUET `
        --geo-profile $GEO_PROFILE `
        --species-map $SPECIES_MAP `
        --geo-mask $GEO_MASK `
        --habitat-mask $HABITAT_MASK `
        --biogeo-npz $BIOGEO_NPZ `
        --refined-ref-json $REFINED_REF `
        --radius-km $HAB_R `
        --beta $HAB_BETA `
        --n-near-min $HAB_N `
        --use-altitude `
        --skip-aquatic `
        --skip-biogeo `
        --threshold $THR `
        --topk $TOPK `
        --n-floor 3 `
        --ratio-floor 0.5 `
        --out-csv $OUT_CSV
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at $tag habitat-fit" -ForegroundColor Red; exit 1 }
    Write-Host "  wrote -> $OUT_CSV"
    Write-Host ""
}

Write-Host "==========================================================="
Write-Host "Done."
foreach ($v in $variants) { Write-Host "  abl4_scale$($v.tag).csv" }
Write-Host "==========================================================="
