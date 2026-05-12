# Test: all 8 sources at alpha=0.20 with sim_w on all (interpolation point).
#
# Reference points so far:
#   Anchor (RNNB alpha=0.30, others 0.15, RNNB-only sim_w)  priv 0.39400
#   All sources alpha=0.30 + sim_w on all                    priv 0.38173  (-0.01227)
#   All sources alpha=0.50 + sim_w on all                    priv 0.38110  (-0.01290)
#
# This run fills the gap between anchor's "0.15 default" and the all-0.30
# regression, asking: does lighter uniform temporal fusion across all
# sources stay below noise?

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

$NOKNN_PREFUSION = "$PACE_DL\result_parquets\abl2_vitL_noknn_prefusion.parquet"
$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"
$TOPK = "10"
$HAB_BETA = "1.0"
$HAB_R = "250"
$HAB_N = "40"

$OUT_DIR = "$ABLN\outputs\tf_sweep_allsrc"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR | Out-Null

# Controller: all 8 sources get default_alpha=0.20 (no per-source overrides)
$ctrl_a020 = "$CFG_DIR\vit_all_a020.json"
Set-Content -Path $ctrl_a020 -Encoding ascii -NoNewline `
    -Value '{"alpha_src": {}, "default_alpha": 0.20, "lambda_c": 0.0, "lambda_d": 0.0, "lambda_s": 0.0, "alpha_min": 0.05, "alpha_max": 0.95}'

# sim_w on all 8 sources (reuse from previous run)
$agg_all = "$CFG_DIR\agg_all_sources.json"
if (-not (Test-Path $agg_all)) {
    Set-Content -Path $agg_all -Encoding ascii -NoNewline `
        -Value '{"RNNB":"sim_w","CBN-PdlC":"sim_w","CBN-Pla":"sim_w","CBN-can":"sim_w","GUARDEN":"sim_w","LISAH":"sim_w","OPTMix":"sim_w","2024-CEV3":"sim_w"}'
}

$TAG = "tf_all_a020_simw_all"

Write-Host "==========================================================="
Write-Host "Variant: $TAG"
Write-Host "  controller: $ctrl_a020  (default_alpha=0.20)"
Write-Host "  aggregator: sim_w on all 8 sources"
Write-Host "==========================================================="

& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $NOKNN_PREFUSION `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-controller $ctrl_a020 `
    --vit-aggregator max `
    --vit-aggregator-per-source $agg_all `
    --sim-json $SIM_JSON `
    --tag $TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at ensemble" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"
if (-not (Test-Path $ENS_PARQUET)) {
    Write-Host "ERROR: missing ensemble parquet at $ENS_PARQUET" -ForegroundColor Red
    exit 1
}

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
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Compare against:"
Write-Host "  anchor 0.39400  (RNNB-only, others alpha=0.15)"
Write-Host "  all alpha=0.30  0.38173"
Write-Host "  all alpha=0.50  0.38110"
