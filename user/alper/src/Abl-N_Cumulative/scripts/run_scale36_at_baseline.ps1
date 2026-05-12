# Test scales 3+6 at the full Baseline downstream (minus kNN).
#
# Setup:
#   ViT-L scales 3+6 (already-built no-kNN prefusion from Abl-4 sweep)
#   Temporal fusion: vit_rnnb030.json controller + {"RNNB":"sim_w"} aggregator
#   Masks: aquatic + biogeo  (Baseline)
#   Habitat-fit: beta=0.10, R=250, n=40, altitude on  (Baseline)
#   Admission: defaults (no per-source overrides)
#   kNN: OFF (we don't have FAISS + ArcFace tiles locally)
#
# Reference points:
#   Baseline (4+5+6+7 with kNN): priv 0.45732 / pub 0.47351
#   Abl-1 (4+5+6+7 minus kNN):   priv 0.44785 / pub 0.45168
#   Abl-3 anchor at 4+5+6+7:      priv 0.38961 / pub 0.41178
#   Abl-4 3+6 at Abl-3 config:    priv 0.40637 / pub 0.39639  (+0.01676 priv vs anchor)
#
# Predicted: this run (3+6 + full downstream, no kNN) should land near
#   0.44785 + 0.01676 = 0.46461 priv  if the 3+6 scale advantage transfers cleanly.
# Projected Baseline-equivalent (adding ~+0.009 kNN contribution):
#   ~0.46461 + 0.009 = ~0.47361 priv

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# Re-use the no-kNN prefusion built during the Abl-4 sweep at scales 3+6
$PREFUSION = "$ABLN\outputs\abl4_multiscale\prefusion\abl4_scale36_prefusion.parquet"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$VIT_CONTROLLER= "$EXP14\stage_configs\controllers\vit_rnnb030.json"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

# Baseline downstream knobs (no admission overrides, masks ON, habitat-fit ON)
$THR = "0.085"; $TOPK = "10"
$HAB_BETA = "0.10"; $HAB_R = "250"; $HAB_N = "40"

$OUT_DIR = "$ABLN\outputs\scale36_baseline"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR | Out-Null

# Aggregator JSON file (PowerShell can't pass JSON literal safely)
$agg = "$CFG_DIR\agg_rnnb_only.json"
Set-Content -Path $agg -Encoding ascii -NoNewline -Value '{"RNNB":"sim_w"}'

$required = @($PREFUSION, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $VIT_CONTROLLER, $SIM_JSON,
              $GEO_PROFILE, $REFINED_REF, $agg)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file: $f" -ForegroundColor Red
        exit 1
    }
}

$TAG = "scale36_baseline_noknn"

Write-Host "==========================================================="
Write-Host "Scale 3+6 at full Baseline downstream (kNN OFF)"
Write-Host "==========================================================="
Write-Host "Prefusion: $PREFUSION"
Write-Host "Targets to compare:"
Write-Host "  Abl-1 (4+5+6+7 minus kNN, full downstream): 0.44785 priv / 0.45168 pub"
Write-Host "  Predicted lift from 3+6 vs 4+5+6+7: +0.01676 priv"
Write-Host "  Naive prediction: 0.44785 + 0.01676 = ~0.46461 priv"
Write-Host ""

# Step B: ensemble with temporal fusion ON (Baseline controller + RNNB sim_w)
& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-controller $VIT_CONTROLLER `
    --vit-aggregator max `
    --vit-aggregator-per-source $agg `
    --sim-json $SIM_JSON `
    --tag $TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at ensemble" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"

# Step C: habitat-fit at full Baseline config (β=0.10, masks ON, no admission)
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
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Compare to Abl-1 (0.44785 priv) -- adds kNN-off downstream"
Write-Host "Add ~+0.009 to project the full Baseline-equivalent with kNN restored."
