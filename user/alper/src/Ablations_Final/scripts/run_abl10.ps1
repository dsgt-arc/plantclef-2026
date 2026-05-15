# Abl-10 (test): bare 3+4+5+6 + kNN only
#
# Adds kNN BACK to the most-stripped Abl-3 config (no temporal fusion, no masks,
# no habitat-fit, no admission). Tests kNN's marginal contribution when it is
# the ONLY mechanism on top of the bare classifier at 3+4+5+6 multi-scale.
#
# Comparison anchors:
#   Abl-3 (bare 3+4+5+6, no kNN):    priv 0.40079 / pub 0.39835
#   Baseline (3+4+5+6 + everything):  priv 0.45777 / pub 0.46914
#
# Expected: kNN contribution ~+0.009 priv (per prior measurements at 4+5+6+7).

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1_SCRIPT_DIR = "$REPO\user\alper\src\Abl-1_CNext\scripts"
$AF_ROOT = "$REPO\user\alper\src\Ablations_Final"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# kNN-enabled prefusion at scales 3+4+5+6 (built during the Baseline 3456 run)
$PREFUSION_WITH_KNN = "$REPO\user\alper\src\Abl-N_Cumulative\outputs\scale_more_baseline_knn\prefusion\vitL_scale3456_prefusion_b0.70.parquet"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"; $TOPK = "10"
$HAB_BETA_OFF = "1.0"
$HAB_R = "250"; $HAB_N = "40"

$ABL10_OUT = "$AF_ROOT\Abl-10_AddKNN_To_Bare\outputs"
$ABL10_TAG = "Abl-10_AddKNN_To_Bare"

$required = @($PREFUSION_WITH_KNN, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $GEO_PROFILE, $REFINED_REF)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Abl-10 -- bare 3+4+5+6 + kNN (no other mechanisms)"
Write-Host "==========================================================="
Write-Host "Compare to:"
Write-Host "  Abl-3 (bare 3+4+5+6, no kNN):    priv 0.40079 / pub 0.39835"
Write-Host "  Baseline (3+4+5+6 + everything):  priv 0.45777 / pub 0.46914"
Write-Host ""

# Step: ensemble with baseline controller (no temporal fusion), kNN already baked into prefusion
& $PYTHON "$ABL1_SCRIPT_DIR\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION_WITH_KNN `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $ABL10_OUT `
    --thresholds $THR `
    --topk $TOPK `
    --vit-aggregator max `
    --tag $ABL10_TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at ensemble" -ForegroundColor Red; exit 1 }

$ABL10_ENS = "$ABL10_OUT\${ABL10_TAG}_w1.00_ensemble.parquet"

# Step: habitat-fit at bare config (beta=1.0, skip both masks)
$ABL10_CSV = "$ABL10_OUT\${ABL10_TAG}.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ABL10_ENS `
    --geo-profile $GEO_PROFILE `
    --species-map $SPECIES_MAP `
    --geo-mask $GEO_MASK `
    --habitat-mask $HABITAT_MASK `
    --biogeo-npz $BIOGEO_NPZ `
    --refined-ref-json $REFINED_REF `
    --radius-km $HAB_R `
    --beta $HAB_BETA_OFF `
    --n-near-min $HAB_N `
    --use-altitude `
    --skip-aquatic `
    --skip-biogeo `
    --threshold $THR `
    --topk $TOPK `
    --n-floor 3 `
    --ratio-floor 0.5 `
    --out-csv $ABL10_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at habitat-fit" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Done. CSV: $ABL10_CSV"
Write-Host "Compare to Abl-3 anchor (0.40079) -- difference = kNN's marginal contribution at bare 3+4+5+6"
