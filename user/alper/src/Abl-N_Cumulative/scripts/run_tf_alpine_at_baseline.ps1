# Test alpine temporal fusion at the Baseline level (masks ON, kNN ON,
# habitat-fit ON, admission OFF).
#
# Hypothesis: in Abl-2 (stripped, no masks) alpine sim_w hurt by ~0.014 because
# alpine predictions were polluted with non-alpine species. At Baseline the
# biogeo mask filters those out, so alpine sim_w on cleaned predictions
# might help instead.
#
# Configuration (one isolated change from Baseline):
#   RNNB     alpha=0.30 + sim_w    (unchanged from Baseline)
#   CBN-PdlC alpha=0.20 + sim_w    (NEW; was alpha=0.15 + max)
#   CBN-Pla  alpha=0.20 + sim_w    (NEW; was alpha=0.15 + max)
#   others   alpha=0.15 + max      (unchanged)
#
# Anchor: Baseline (final paper pipeline) -> priv 0.45732 / pub 0.47351
#
# Usage:  pwsh .\run_tf_alpine_at_baseline.ps1

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# Use the WITH-kNN prefusion parquet for Baseline-level test
$PREFUSION_WITH_KNN = "$PACE_DL\result_parquets\vitL_prefusion_b0.70.parquet"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

# Baseline-locked params (everything matches Baseline except the controller/aggregator)
$THR = "0.085"
$TOPK = "10"
$HAB_BETA = "0.10"     # habitat-fit ON
$HAB_R = "250"
$HAB_N = "40"

$OUT_DIR = "$ABLN\outputs\tf_alpine_baseline"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR | Out-Null

# Custom controller: RNNB at 0.30, alpine at 0.20, default 0.15
$ctrl = "$CFG_DIR\vit_rnnb030_alpine020.json"
Set-Content -Path $ctrl -Encoding ascii -NoNewline `
    -Value '{"alpha_src": {"RNNB": 0.30, "CBN-PdlC": 0.20, "CBN-Pla": 0.20}, "default_alpha": 0.15, "lambda_c": 0.0, "lambda_d": 0.0, "lambda_s": 0.0, "alpha_min": 0.05, "alpha_max": 0.95}'

# sim_w on the three multi-visit sources
$agg = "$CFG_DIR\agg_multivisit3.json"
Set-Content -Path $agg -Encoding ascii -NoNewline `
    -Value '{"RNNB":"sim_w","CBN-PdlC":"sim_w","CBN-Pla":"sim_w"}'

$TAG = "tf_alpine_baseline_a020"

$required = @($PREFUSION_WITH_KNN, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $SIM_JSON, $GEO_PROFILE, $REFINED_REF, $ctrl, $agg)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Alpine fusion test at Baseline level: $TAG"
Write-Host "==========================================================="
Write-Host "Anchor (Baseline, RNNB-only sim_w): priv 0.45732 / pub 0.47351"
Write-Host "Change: add sim_w + alpha=0.20 on CBN-PdlC, CBN-Pla"
Write-Host "Held: kNN on, masks on, habitat-fit on, admission off"
Write-Host ""

# Step A: ensemble with kNN prefusion + new controller + alpine sim_w
& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION_WITH_KNN `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-controller $ctrl `
    --vit-aggregator max `
    --vit-aggregator-per-source $agg `
    --sim-json $SIM_JSON `
    --tag $TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at ensemble" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"
if (-not (Test-Path $ENS_PARQUET)) {
    Write-Host "ERROR: missing ensemble parquet at $ENS_PARQUET" -ForegroundColor Red
    exit 1
}

# Step B: habitat-fit at Baseline config (masks on, admission off)
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
    --threshold $THR `
    --topk $TOPK `
    --n-floor 3 `
    --ratio-floor 0.5 `
    --out-csv $OUT_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at habitat-fit" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "==========================================================="
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Compare to Baseline 0.45732 priv / 0.47351 pub"
Write-Host "==========================================================="
