# Multi-scale variant: 3+5+6  (70 tiles/image).
# Skip-pattern combo -- tests whether dropping scale 4 from {3,4,5,6} hurts.

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

$THR = "0.085"; $TOPK = "10"
$HAB_BETA = "1.0"; $HAB_R = "250"; $HAB_N = "40"

$OUT_DIR  = "$ABLN\outputs\abl4_multiscale"
$PREF_DIR = "$OUT_DIR\prefusion"
New-Item -Force -ItemType Directory $OUT_DIR  | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR | Out-Null

$tag = "abl4_scale356"
$scale_paths = @(
    "$SCALE_DIR\scale_3x3.parquet",
    "$SCALE_DIR\scale_5x5.parquet",
    "$SCALE_DIR\scale_6x6.parquet"
)

Write-Host "==========================================================="
Write-Host "VARIANT: $tag  -- scales: 3,5,6  (70 tiles/image)"
Write-Host "==========================================================="

& $PYTHON "$ABL2\scripts\make_vitL_prefusion_no_knn.py" `
    --vit-scales $scale_paths `
    --class-mask-npy $GEO_MASK `
    --out-dir $PREF_DIR `
    --temperature 1.5 `
    --image-topk 64 `
    --tag $tag
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at prefusion" -ForegroundColor Red; exit 1 }

$PREFUSION = "$PREF_DIR\${tag}_prefusion.parquet"

& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-aggregator max `
    --tag $tag
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at ensemble" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${tag}_w1.00_ensemble.parquet"
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
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at habitat-fit" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $OUT_CSV"
