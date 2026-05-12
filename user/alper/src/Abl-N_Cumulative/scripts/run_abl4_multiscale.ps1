# Multi-scale sweep (NEW Abl-4).
#
# All variants run at the Abl-3 cumulative config:
#   - no kNN (uses local phase2v3 per-tile parquets, no FAISS)
#   - no temporal fusion (baseline controller, max aggregator, no sim_w)
#   - no habitat-fit (beta=1.0)
#   - no aquatic / biogeo masks (--skip-aquatic --skip-biogeo)
#   - no per-source admission overrides
#
# Variants (8 total):
#   3   -- scale 3x3 only             ( 9 tiles/image)
#   4   -- scale 4x4 only             (16)
#   5   -- scale 5x5 only             (25)
#   6   -- scale 6x6 only             (36)
#   345 -- scales 3+4+5               (50)
#   456 -- scales 4+5+6               (77)
#   4567-- scales 4+5+6+7             (126)  <- matches Abl-3 anchor (0.38961 priv)
#   45678 - scales 4+5+6+7+8          (190)
#
# Reference anchor (Abl-3, scales 4+5+6+7): priv 0.38961 / pub 0.41178

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

# Helper: scale paths
function Scales([int[]]$scales) {
    return $scales | ForEach-Object { "$SCALE_DIR\scale_${_}x${_}.parquet" }
}

$variants = @(
    @{ tag = "3";     scales = @(3) },
    @{ tag = "4";     scales = @(4) },
    @{ tag = "5";     scales = @(5) },
    @{ tag = "6";     scales = @(6) },
    @{ tag = "345";   scales = @(3,4,5) },
    @{ tag = "456";   scales = @(4,5,6) },
    @{ tag = "4567";  scales = @(4,5,6,7) },
    @{ tag = "45678"; scales = @(4,5,6,7,8) }
)

# Pre-flight: every required scale parquet exists locally
$all_scales_needed = ($variants | ForEach-Object { $_.scales } | Sort-Object -Unique)
foreach ($s in $all_scales_needed) {
    $p = "$SCALE_DIR\scale_${s}x${s}.parquet"
    if (-not (Test-Path $p)) {
        Write-Host "ERROR: missing local scale parquet: $p" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Abl-4 multi-scale sweep"
Write-Host "==========================================================="
Write-Host "Anchor (4+5+6+7, Abl-3): priv 0.38961 / pub 0.41178"
Write-Host ""

foreach ($v in $variants) {
    $tag = "abl4_scale$($v.tag)"
    Write-Host "==========================================================="
    Write-Host "VARIANT: $tag  -- scales: $($v.scales -join ',')"
    Write-Host "==========================================================="

    $scale_paths = Scales $v.scales

    # Step A: no-kNN prefusion
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

    # Step B: ensemble (baseline controller -- no temporal fusion)
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

    # Step C: habitat-fit (Abl-3 cumulative config: beta=1.0, skip masks, no admission)
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
Write-Host "Done. CSVs in $OUT_DIR :"
foreach ($v in $variants) { Write-Host "  abl4_scale$($v.tag).csv" }
Write-Host ""
Write-Host "Sanity check: abl4_scale4567.csv should match Abl-3 anchor (0.38961 priv)."
Write-Host "==========================================================="
