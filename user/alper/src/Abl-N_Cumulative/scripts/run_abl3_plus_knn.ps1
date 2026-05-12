# Test: Abl-3 (no temporal fusion, no masks, no habitat-fit, no admission)
#       PLUS kNN turned back on.
#
# Tests whether the kNN marginal contribution depends on temporal fusion
# being active in the pipeline.
#
# Reference points:
#   Abl-3 (no kNN, no temporal):    priv 0.38961 / pub 0.41178   <- anchor
#   Abl-2 (no kNN, WITH temporal):  priv 0.39400 / pub 0.42184
#   Abl-1 (kNN, WITH temporal):     priv 0.44785 / pub 0.45168
#   Baseline (kNN + temporal + masks + habitat-fit + admission off): 0.45732
#
# Configuration:
#   Input prefusion: vitL_prefusion_b0.70.parquet (kNN baked in at beta_match=0.70)
#   Controller: baseline (no per-source alpha) -- no temporal fusion
#   Aggregator: max (no sim_w) -- no temporal fusion
#   Habitat-fit: beta=1.0 (identity, no demotion)
#   Masks: --skip-aquatic --skip-biogeo (off)
#   Admission: no --per-source-json (defaults only)

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# Prefusion WITH kNN baked in
$PREFUSION_WITH_KNN = "$PACE_DL\result_parquets\vitL_prefusion_b0.70.parquet"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"
$TOPK = "10"
$HAB_BETA = "1.0"    # identity, no demotion
$HAB_R = "250"
$HAB_N = "40"

$OUT_DIR = "$ABLN\outputs\abl3_plus_knn"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null

$TAG = "abl3_plus_knn"

$required = @($PREFUSION_WITH_KNN, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $GEO_PROFILE, $REFINED_REF)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Abl-3 + kNN test"
Write-Host "==========================================================="
Write-Host "Anchor (Abl-3, no kNN, no temporal): priv 0.38961 / pub 0.41178"
Write-Host "Adding back: kNN at beta_match=0.70"
Write-Host "Still off: temporal fusion, masks, habitat-fit, admission, ConvNeXt"
Write-Host ""

# Step A: ensemble with kNN prefusion + BASELINE controller (no temporal fusion)
& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION_WITH_KNN `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-aggregator max `
    --tag $TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at ensemble" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"
if (-not (Test-Path $ENS_PARQUET)) {
    Write-Host "ERROR: missing ensemble parquet at $ENS_PARQUET" -ForegroundColor Red
    exit 1
}

# Step B: habitat-fit at Abl-3 cumulative config
$OUT_CSV = "$OUT_DIR\${TAG}.csv"
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
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at habitat-fit" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "==========================================================="
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Compare to Abl-3 anchor 0.38961 / 0.41178"
Write-Host "kNN's marginal contribution = (this result) - 0.38961"
Write-Host "==========================================================="
