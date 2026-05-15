# Re-run Abl-1, Abl-2, Abl-3 at the new Baseline: ViT-L + 3+4+5+6 multi-scale.
#
# New Baseline (already measured): scale3456_baseline_knn.csv
#   priv 0.45777 / pub 0.46914
#   Final pipeline: ViT-L + 3+4+5+6 + kNN + temporal fusion + masks + habitat-fit,
#                   no admission overrides, no ConvNeXt.
#
# Cumulative chain:
#   Abl-1: + kNN retrieval blend            -> drop kNN, keep everything else
#   Abl-2: + habitat-fit demotion + masks   -> drop hab-fit (beta=1.0) + skip masks
#   Abl-3: + temporal fusion                -> drop temporal fusion (baseline ctrl)
#
# Note: Abl-4 (multi-scale) and Abl-5 (ViT-B) are unchanged from the prior runs
# and not re-executed here.
#
# All three runs reuse the existing no-kNN prefusion at scales 3+4+5+6:
#   Abl-N_Cumulative/outputs/abl4_multiscale/prefusion/abl4_scale3456_prefusion.parquet
# (Built during the Abl-4 multi-scale sweep -- no kNN, no temporal fusion baked in.)

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1_SCRIPT_DIR = "$REPO\user\alper\src\Abl-1_CNext\scripts"
$AF_ROOT = "$REPO\user\alper\src\Ablations_Final"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# Existing no-kNN prefusion at scales 3+4+5+6 (built during Abl-4 multi-scale sweep)
$PREFUSION_NO_KNN = "$REPO\user\alper\src\Abl-N_Cumulative\outputs\abl4_multiscale\prefusion\abl4_scale3456_prefusion.parquet"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$VIT_CONTROLLER= "$EXP14\stage_configs\controllers\vit_rnnb030.json"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

# Baseline-locked knobs (no admission overrides at the new Baseline)
$THR = "0.085"; $TOPK = "10"
$HAB_BETA_ON  = "0.10"
$HAB_BETA_OFF = "1.0"
$HAB_R = "250"; $HAB_N = "40"

# Pre-flight
$required = @($PREFUSION_NO_KNN, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $VIT_CONTROLLER, $SIM_JSON, $GEO_PROFILE, $REFINED_REF)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Re-running Abl-1, Abl-2, Abl-3 at the new Baseline (3+4+5+6)"
Write-Host "==========================================================="
Write-Host "New Baseline: ViT-L + 3+4+5+6 + kNN + temporal fusion + masks + habitat-fit, no admission, no ConvNeXt"
Write-Host "Anchor: priv 0.45777 / pub 0.46914"
Write-Host ""

# ============================================================
# Abl-1 -- drop kNN retrieval blend (everything else identical to Baseline)
# ============================================================
Write-Host "==========================================================="
Write-Host "Abl-1 -- drop kNN retrieval blend"
Write-Host "==========================================================="
$ABL1_OUT = "$AF_ROOT\Abl-1_NoKNN\outputs"
$ABL1_CFG = "$AF_ROOT\Abl-1_NoKNN\configs"
New-Item -Force -ItemType Directory $ABL1_CFG | Out-Null
$AGG_RNNB = "$ABL1_CFG\agg_rnnb_only.json"
Set-Content -Path $AGG_RNNB -Encoding ascii -NoNewline -Value '{"RNNB":"sim_w"}'

# Step: ensemble with vit_rnnb030 + RNNB sim_w (temporal fusion ON, no kNN)
$ABL1_TAG = "Abl-1_NoKNN"
& $PYTHON "$ABL1_SCRIPT_DIR\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION_NO_KNN `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $ABL1_OUT `
    --thresholds $THR `
    --topk $TOPK `
    --vit-controller $VIT_CONTROLLER `
    --vit-aggregator max `
    --vit-aggregator-per-source $AGG_RNNB `
    --sim-json $SIM_JSON `
    --tag $ABL1_TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Abl-1 ensemble" -ForegroundColor Red; exit 1 }

$ABL1_ENS = "$ABL1_OUT\${ABL1_TAG}_w1.00_ensemble.parquet"

# Step: habitat-fit with masks ON + beta=0.10 + no admission
$ABL1_CSV = "$ABL1_OUT\${ABL1_TAG}.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ABL1_ENS `
    --geo-profile $GEO_PROFILE `
    --species-map $SPECIES_MAP `
    --geo-mask $GEO_MASK `
    --habitat-mask $HABITAT_MASK `
    --biogeo-npz $BIOGEO_NPZ `
    --refined-ref-json $REFINED_REF `
    --radius-km $HAB_R `
    --beta $HAB_BETA_ON `
    --n-near-min $HAB_N `
    --use-altitude `
    --threshold $THR `
    --topk $TOPK `
    --n-floor 3 `
    --ratio-floor 0.5 `
    --out-csv $ABL1_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Abl-1 habitat-fit" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $ABL1_CSV"
Write-Host ""

# ============================================================
# Abl-2 -- drop habitat-fit demotion + skip aquatic + skip biogeo
# Cumulative: same ensemble parquet as Abl-1 (no kNN, temporal fusion ON).
# ============================================================
Write-Host "==========================================================="
Write-Host "Abl-2 -- drop habitat-fit demotion + masks (cumulative on Abl-1)"
Write-Host "==========================================================="
$ABL2_OUT = "$AF_ROOT\Abl-2_NoHabFit\outputs"
$ABL2_TAG = "Abl-2_NoHabFit"
$ABL2_CSV = "$ABL2_OUT\${ABL2_TAG}.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ABL1_ENS `
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
    --out-csv $ABL2_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Abl-2 habitat-fit" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $ABL2_CSV"
Write-Host ""

# ============================================================
# Abl-3 -- drop temporal fusion (baseline controller, plain max aggregator)
# Cumulative: drops kNN, hab-fit, masks, temporal fusion. Bare ViT-L + 3+4+5+6.
# ============================================================
Write-Host "==========================================================="
Write-Host "Abl-3 -- drop temporal fusion (cumulative on Abl-2)"
Write-Host "==========================================================="
$ABL3_OUT = "$AF_ROOT\Abl-3_NoTemporal\outputs"
$ABL3_TAG = "Abl-3_NoTemporal"

# New ensemble step with baseline controller (no temporal fusion)
& $PYTHON "$ABL1_SCRIPT_DIR\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION_NO_KNN `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $ABL3_OUT `
    --thresholds $THR `
    --topk $TOPK `
    --vit-aggregator max `
    --tag $ABL3_TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Abl-3 ensemble" -ForegroundColor Red; exit 1 }

$ABL3_ENS = "$ABL3_OUT\${ABL3_TAG}_w1.00_ensemble.parquet"

# habitat-fit with beta=1.0 + skip masks
$ABL3_CSV = "$ABL3_OUT\${ABL3_TAG}.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ABL3_ENS `
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
    --out-csv $ABL3_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Abl-3 habitat-fit" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $ABL3_CSV"
Write-Host ""

Write-Host "==========================================================="
Write-Host "Done. CSVs:"
Write-Host "  Baseline (already measured) : Ablations_Final\Baseline\Baseline_vitL_3456.csv  -- priv 0.45777 / pub 0.46914"
Write-Host "  Abl-1 (no kNN)              : $ABL1_CSV"
Write-Host "  Abl-2 (no hab-fit + masks)  : $ABL2_CSV"
Write-Host "  Abl-3 (no temporal fusion)  : $ABL3_CSV"
Write-Host "==========================================================="
