# ViT-B multi-scale ablation (mirror of the ViT-L Abl-4 multi-scale sweep).
#
# Same Abl-3 cumulative config for direct apples-to-apples comparison with
# the ViT-L sweep:
#   - no kNN
#   - no temporal fusion (baseline controller, plain max aggregator)
#   - no habitat-fit demotion (beta=1.0)
#   - no aquatic / biogeo masks (--skip-aquatic --skip-biogeo)
#   - no per-source admission overrides
#
# Variants (7 total):
#   3, 4, 5, 6      -- single scales
#   456             -- 3-scale (4+5+6)
#   3456            -- 4-scale (3+4+5+6, the final ViT-L pipeline multi-scale)
#   4567            -- 4-scale (4+5+6+7, the production scheme)
#
# ViT-B per-tile parquets:
#   PACE_downloads/vitB_multiscale/exp7_multiscale/scale_NxN.parquet
#   Contains image_name + teacher_logits_f16 (classifier-only, 7806-class output).

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABL2  = "$REPO\user\alper\src\Abl-2_KNN"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"
$SCALE_DIR = "$PACE_DL\vitB_multiscale\exp7_multiscale"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"; $TOPK = "10"
$HAB_BETA = "1.0"; $HAB_R = "250"; $HAB_N = "40"

$OUT_DIR  = "$ABLN\outputs\vitB_multiscale"
$PREF_DIR = "$OUT_DIR\prefusion"
New-Item -Force -ItemType Directory $OUT_DIR  | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR | Out-Null

function Scales([int[]]$scales) {
    return $scales | ForEach-Object { "$SCALE_DIR\scale_${_}x${_}.parquet" }
}

$variants = @(
    @{ tag = "3";    scales = @(3) },
    @{ tag = "4";    scales = @(4) },
    @{ tag = "5";    scales = @(5) },
    @{ tag = "6";    scales = @(6) },
    @{ tag = "456";  scales = @(4,5,6) },
    @{ tag = "3456"; scales = @(3,4,5,6) },
    @{ tag = "4567"; scales = @(4,5,6,7) }
)

# Pre-flight: verify every required scale parquet exists locally
$all_scales_needed = ($variants | ForEach-Object { $_.scales } | Sort-Object -Unique)
foreach ($s in $all_scales_needed) {
    $p = "$SCALE_DIR\scale_${s}x${s}.parquet"
    if (-not (Test-Path $p)) {
        Write-Host "ERROR: missing local ViT-B scale parquet: $p" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "ViT-B multi-scale ablation (Abl-3 cumulative config)"
Write-Host "==========================================================="
Write-Host "ViT-L comparators (same level):"
Write-Host "  scale 3:        priv 0.38533 / pub 0.33787"
Write-Host "  scale 4:        priv 0.37920 / pub 0.35106"
Write-Host "  scale 5:        priv 0.38492 / pub 0.36886"
Write-Host "  scale 6:        priv 0.39332 / pub 0.39770"
Write-Host "  4+5+6:          priv 0.39317 / pub 0.39882"
Write-Host "  3+4+5+6:        priv 0.40079 / pub 0.39835"
Write-Host "  4+5+6+7:        priv 0.38961 / pub 0.41178"
Write-Host ""

foreach ($v in $variants) {
    $tag = "vitB_scale$($v.tag)"
    Write-Host "==========================================================="
    Write-Host "VARIANT: $tag  -- scales: $($v.scales -join ',')"
    Write-Host "==========================================================="

    $scale_paths = Scales $v.scales

    # Step A: no-kNN prefusion (uses the backbone-agnostic Abl-2 script)
    & $PYTHON "$ABL2\scripts\make_vitL_prefusion_no_knn.py" `
        --vit-scales $scale_paths `
        --class-mask-npy $GEO_MASK `
        --out-dir $PREF_DIR `
        --temperature 1.5 `
        --image-topk 64 `
        --tag $tag
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at $tag prefusion" -ForegroundColor Red; exit 1 }

    $PREFUSION = "$PREF_DIR\${tag}_prefusion.parquet"

    # Step B: ensemble with baseline controller (no temporal fusion)
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

    # Step C: habitat-fit at Abl-3 cumulative (beta=1.0, skip masks)
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
foreach ($v in $variants) { Write-Host "  vitB_scale$($v.tag).csv" }
Write-Host "==========================================================="
