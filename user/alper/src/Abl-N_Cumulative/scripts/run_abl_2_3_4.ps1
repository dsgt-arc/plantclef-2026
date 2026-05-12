# Cumulative ablations 2, 3, 4 (new numbering -- see Ablation order.docx).
#
# Anchor: "Baseline" = Abl-1 from old numbering, i.e. ViT-L + kNN + masks +
#         habitat-fit + admission (no ConvNeXt). Private 0.45001 / public 0.47136.
#
# Cumulative chain:
#   Anchor               priv 0.45001  (Baseline final pipeline)
#     - kNN              priv 0.44053  (OLD Abl-2 / NEW Abl-1, done)
#       - admission      priv ?         (NEW Abl-2, this script)
#         - habitat-fit  priv ?         (NEW Abl-3, beta=1.0 + skip aquatic + skip biogeo)
#           - temporal   priv ?         (NEW Abl-4, baseline controller, no sim_w)
#
# All runs operate on the existing Abl-1 ensemble parquet, except Abl-4
# which re-runs the ensemble step with the baseline controller.
#
# Usage:  pwsh .\run_abl_2_3_4.ps1

$ErrorActionPreference = "Stop"

$PYTHON = "C:\Users\alper\anaconda3\python.exe"

# ---- Roots ----
$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABL2  = "$REPO\user\alper\src\Abl-2_KNN"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"

$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

# ---- Inputs ----
$ENS_PARQUET_BASELINE = "$PACE_DL\result_parquets\abl2_vitL_noknn_w1.00_ensemble.parquet"
$NOKNN_PREFUSION      = "$PACE_DL\result_parquets\abl2_vitL_noknn_prefusion.parquet"
$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$VIT_CONTROLLER= "$EXP14\stage_configs\controllers\vit_rnnb030.json"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

# ---- Locked params (B_R250 lock, held identical to Abl-1) ----
$THR     = "0.085"
$TOPK    = "10"
$HAB_BETA = "0.10"
$HAB_R   = "250"
$HAB_N   = "40"

$OUT_DIR = "$ABLN\outputs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null

# Pre-flight: check all required files exist
$required = @($ENS_PARQUET_BASELINE, $NOKNN_PREFUSION,
              $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK, $BIOGEO_NPZ,
              $VIT_CONTROLLER, $SIM_JSON, $GEO_PROFILE, $REFINED_REF)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Cumulative ablations: NEW Abl-2 / Abl-3 / Abl-4"
Write-Host "==========================================================="
Write-Host "Anchor (Baseline, final pipeline):"
Write-Host "  priv 0.45001 / pub 0.47136"
Write-Host "Cumulative previous step (Abl-1, kNN off):"
Write-Host "  priv 0.44053 / pub 0.44953"
Write-Host ""

# ============================================================
# NEW Abl-2 -- drop per-source admission overrides
# ============================================================
Write-Host "==========================================================="
Write-Host "NEW Abl-2 -- drop per-source admission overrides"
Write-Host "==========================================================="
$ABL2_CSV = "$OUT_DIR\new_abl2_no_admission.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ENS_PARQUET_BASELINE `
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
    --out-csv $ABL2_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at NEW Abl-2" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $ABL2_CSV"
Write-Host ""

# ============================================================
# NEW Abl-3 -- drop habitat-fit demotion (lumped: beta=1.0 + skip aquatic + skip biogeo)
# Cumulative from Abl-2 (no admission overrides).
# ============================================================
Write-Host "==========================================================="
Write-Host "NEW Abl-3 -- drop habitat-fit demotion + aquatic + biogeo masks"
Write-Host "==========================================================="
$ABL3_CSV = "$OUT_DIR\new_abl3_no_habitat_filters.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ENS_PARQUET_BASELINE `
    --geo-profile $GEO_PROFILE `
    --species-map $SPECIES_MAP `
    --geo-mask $GEO_MASK `
    --habitat-mask $HABITAT_MASK `
    --biogeo-npz $BIOGEO_NPZ `
    --refined-ref-json $REFINED_REF `
    --radius-km $HAB_R `
    --beta 1.0 `
    --n-near-min $HAB_N `
    --use-altitude `
    --skip-aquatic `
    --skip-biogeo `
    --threshold $THR `
    --topk $TOPK `
    --n-floor 3 `
    --ratio-floor 0.5 `
    --out-csv $ABL3_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at NEW Abl-3" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $ABL3_CSV"
Write-Host ""

# ============================================================
# NEW Abl-4 -- drop temporal fusion (baseline controller, no sim_w on RNNB)
# Re-runs ensemble step with baseline ControllerConfig and no per-source aggregator,
# then apply_habitat_fit cumulative (no admission, no habitat-fit demotion, no masks).
# ============================================================
Write-Host "==========================================================="
Write-Host "NEW Abl-4 -- drop temporal fusion (baseline controller + plain max)"
Write-Host "==========================================================="
$ABL4_TAG = "new_abl4_no_temporal"
& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $NOKNN_PREFUSION `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-aggregator max `
    --tag $ABL4_TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at NEW Abl-4 ensemble step" -ForegroundColor Red; exit 1 }

$ABL4_ENS = "$OUT_DIR\${ABL4_TAG}_w1.00_ensemble.parquet"
if (-not (Test-Path $ABL4_ENS)) {
    Write-Host "ERROR: missing ensemble parquet at $ABL4_ENS" -ForegroundColor Red
    exit 1
}

$ABL4_CSV = "$OUT_DIR\new_abl4_no_temporal_fusion.csv"
& $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
    --ensemble-parquet $ABL4_ENS `
    --geo-profile $GEO_PROFILE `
    --species-map $SPECIES_MAP `
    --geo-mask $GEO_MASK `
    --habitat-mask $HABITAT_MASK `
    --biogeo-npz $BIOGEO_NPZ `
    --refined-ref-json $REFINED_REF `
    --radius-km $HAB_R `
    --beta 1.0 `
    --n-near-min $HAB_N `
    --use-altitude `
    --skip-aquatic `
    --skip-biogeo `
    --threshold $THR `
    --topk $TOPK `
    --n-floor 3 `
    --ratio-floor 0.5 `
    --out-csv $ABL4_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at NEW Abl-4 habitat-fit step" -ForegroundColor Red; exit 1 }
Write-Host "  wrote -> $ABL4_CSV"

Write-Host ""
Write-Host "==========================================================="
Write-Host "All three ablations done. CSVs:"
Write-Host "  NEW Abl-2 (no admission)       : $ABL2_CSV"
Write-Host "  NEW Abl-3 (no habitat filters) : $ABL3_CSV"
Write-Host "  NEW Abl-4 (no temporal fusion) : $ABL4_CSV"
Write-Host "==========================================================="
