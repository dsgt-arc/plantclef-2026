# Test: alpha=0.20 + sim_w on the THREE genuinely multi-visit sources only.
#
# Multi-visit sources (identified from visit_similarity.json):
#   RNNB      141 entries, avg ~2 neighbors per quadrat
#   CBN-PdlC  816 entries, ~22 neighbors per quadrat (alpine, repeat monitoring)
#   CBN-Pla   628 entries, ~19 neighbors per quadrat (alpine, repeat monitoring)
#
# Single-visit (or sparse) sources -- no temporal fusion applied:
#   CBN-can, GUARDEN, LISAH, OPTMix, 2024-CEV3
#
# Controller: alpha_src = {RNNB: 0.20, CBN-PdlC: 0.20, CBN-Pla: 0.20},
#             default_alpha = 0.0, alpha_min = 0.0 (truly no fusion on others)
# Aggregator: sim_w on the same three; max on the rest.
#
# Reference points (Abl-2 level):
#   Anchor (RNNB-only alpha=0.30, RNNB-only sim_w): priv 0.39400 / pub 0.42184
#   All-source alpha=0.20 + all sim_w (regression):  priv (just submitted) / pub ?
#   All-source alpha=0.30 + all sim_w:               priv 0.38173 / pub 0.38705
#
# Usage:  pwsh .\run_tf_multivisit_a020.ps1

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

$OUT_DIR = "$ABLN\outputs\tf_sweep_multivisit"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR | Out-Null

# Controller: three multi-visit sources at alpha=0.20, everything else at 0 (alpha_min=0 to actually allow zero)
$ctrl_mv = "$CFG_DIR\vit_multivisit_a020.json"
Set-Content -Path $ctrl_mv -Encoding ascii -NoNewline `
    -Value '{"alpha_src": {"RNNB": 0.20, "CBN-PdlC": 0.20, "CBN-Pla": 0.20}, "default_alpha": 0.0, "lambda_c": 0.0, "lambda_d": 0.0, "lambda_s": 0.0, "alpha_min": 0.0, "alpha_max": 0.95}'

# sim_w aggregator scoped to the same three sources
$agg_mv = "$CFG_DIR\agg_multivisit.json"
Set-Content -Path $agg_mv -Encoding ascii -NoNewline `
    -Value '{"RNNB":"sim_w","CBN-PdlC":"sim_w","CBN-Pla":"sim_w"}'

$TAG = "tf_multivisit_a020"

$required = @($NOKNN_PREFUSION, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $SIM_JSON, $GEO_PROFILE, $REFINED_REF, $ctrl_mv, $agg_mv)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Multi-visit-only temporal fusion: $TAG"
Write-Host "==========================================================="
Write-Host "Multi-visit sources at alpha=0.20 + sim_w: RNNB, CBN-PdlC, CBN-Pla"
Write-Host "Single-visit sources: NO temporal fusion (default_alpha=0)"
Write-Host ""

& $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $NOKNN_PREFUSION `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-controller $ctrl_mv `
    --vit-aggregator max `
    --vit-aggregator-per-source $agg_mv `
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
Write-Host "==========================================================="
Write-Host "Done. CSV: $OUT_CSV"
Write-Host ""
Write-Host "Compare:"
Write-Host "  Anchor (RNNB-only alpha=0.30):           priv 0.39400 / pub 0.42184"
Write-Host "  All-source alpha=0.30 (regression):      priv 0.38173 / pub 0.38705"
Write-Host "  All-source alpha=0.50 (regression):      priv 0.38110 / pub 0.38809"
Write-Host "  THIS RUN (multivisit-3 only, alpha=0.20): submit and compare"
Write-Host "==========================================================="
