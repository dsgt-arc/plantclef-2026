# Reordered cumulative test: drop admission BEFORE kNN.
#
# Original cumulative chain (run already):
#   Baseline       priv 0.45001  (kNN on, admission on)
#   - kNN          priv 0.44053  (kNN OFF, admission on)        <- Abl-1
#   - admission    priv 0.44785  (kNN OFF, admission OFF)       <- Abl-2 cumulative
#
# Reordered chain (this script tests the missing point):
#   Baseline       priv 0.45001  (kNN on, admission on)
#   - admission    priv ?         (kNN ON, admission OFF)       <- THIS RUN
#   - kNN          priv 0.44785  (kNN OFF, admission OFF)       <- same as Abl-2
#
# Tells us whether admission and kNN INTERACT.
# - If admission-first single knockout >= Baseline: admission was a public-LB
#   artifact regardless of kNN.
# - If admission-first single knockout < Baseline: admission helps when kNN
#   is present but hurts when kNN is absent. They interact.
#
# Operates on the Abl-1 ensemble parquet (with kNN). Uses the same
# B_R250 habitat-fit lock but drops --per-source-json.
#
# Usage:  pwsh .\run_reordered_admission_first.ps1

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# Use the Abl-1 ensemble parquet (with kNN, no ConvNeXt) -- the Baseline file.
$ENS_PARQUET = "$PACE_DL\result_parquets\abl1_simw_b0.70_w1.00_ensemble.parquet"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"
$TOPK = "10"
$HAB_BETA = "0.10"
$HAB_R = "250"
$HAB_N = "40"

$OUT_DIR = "$ABLN\outputs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null

$required = @($ENS_PARQUET, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $GEO_PROFILE, $REFINED_REF)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Reordered cumulative: drop admission FIRST (keep kNN)"
Write-Host "==========================================================="
Write-Host "Comparison points:"
Write-Host "  Baseline (kNN on, admission on)             priv 0.45001"
Write-Host "  Abl-2 (kNN OFF + admission OFF, cumulative) priv 0.44785"
Write-Host "  THIS RUN (kNN ON, admission OFF)            priv ?"
Write-Host ""

$OUT_CSV = "$OUT_DIR\new_admission_first_no_admission.csv"
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
    --threshold $THR `
    --topk $TOPK `
    --n-floor 3 `
    --ratio-floor 0.5 `
    --out-csv $OUT_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "==========================================================="
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Submit to Kaggle and report private + public F1."
Write-Host "==========================================================="
