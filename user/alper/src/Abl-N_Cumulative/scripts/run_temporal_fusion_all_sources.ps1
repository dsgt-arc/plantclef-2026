# Light tuning: extend TEMPORAL FUSION application to ALL 8 sources.
#
# Previously, RNNB was the only source with explicit alpha override (0.30) and
# the only source with sim_w aggregator. The remaining 7 sources used the
# default_alpha=0.15 controller and the plain max aggregator.
#
# This sweep bumps default_alpha to the higher value and extends sim_w to
# all sources. (sim_w on sources without visit-similarity data degrades to
# max gracefully, so it's safe.)
#
# Anchor (Abl-2 level, vit_rnnb030.json + RNNB:sim_w only):
#   priv 0.39400 / pub 0.42184
#
# Variants:
#   all_a030_simw_all  -- ALL sources alpha=0.30, ALL sources sim_w
#   all_a050_simw_all  -- ALL sources alpha=0.50, ALL sources sim_w
#
# Usage:  pwsh .\run_temporal_fusion_all_sources.ps1

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

# Two custom controllers (default_alpha applies to ALL sources since alpha_src is empty)
$ctrl_a030 = "$CFG_DIR\vit_all_a030.json"
$ctrl_a050 = "$CFG_DIR\vit_all_a050.json"
Set-Content -Path $ctrl_a030 -Encoding ascii -NoNewline `
    -Value '{"alpha_src": {}, "default_alpha": 0.30, "lambda_c": 0.0, "lambda_d": 0.0, "lambda_s": 0.0, "alpha_min": 0.05, "alpha_max": 0.95}'
Set-Content -Path $ctrl_a050 -Encoding ascii -NoNewline `
    -Value '{"alpha_src": {}, "default_alpha": 0.50, "lambda_c": 0.0, "lambda_d": 0.0, "lambda_s": 0.0, "alpha_min": 0.05, "alpha_max": 0.95}'

# sim_w aggregator extended to all 8 sources
$agg_all = "$CFG_DIR\agg_all_sources.json"
Set-Content -Path $agg_all -Encoding ascii -NoNewline `
    -Value '{"RNNB":"sim_w","CBN-PdlC":"sim_w","CBN-Pla":"sim_w","CBN-can":"sim_w","GUARDEN":"sim_w","LISAH":"sim_w","OPTMix":"sim_w","2024-CEV3":"sim_w"}'

$variants = @(
    @{ tag = "tf_all_a030_simw_all"; controller = $ctrl_a030; agg = $agg_all },
    @{ tag = "tf_all_a050_simw_all"; controller = $ctrl_a050; agg = $agg_all }
)

$required = @($NOKNN_PREFUSION, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $SIM_JSON, $GEO_PROFILE, $REFINED_REF,
              $ctrl_a030, $ctrl_a050, $agg_all)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Temporal fusion sweep: ALL sources at the Abl-2 level"
Write-Host "==========================================================="
Write-Host "Anchor (RNNB-only): priv 0.39400 / pub 0.42184"
Write-Host ""

foreach ($v in $variants) {
    Write-Host "==========================================================="
    Write-Host "VARIANT: $($v.tag)"
    Write-Host "  controller: $($v.controller)"
    Write-Host "  aggregator: $($v.agg)"
    Write-Host "==========================================================="

    & $PYTHON "$ABL1\scripts\ensemble_vit_only.py" `
        --vit-prefusion-parquet $NOKNN_PREFUSION `
        --species-map $SPECIES_MAP `
        --class-mask-npy $GEO_MASK `
        --out-dir $OUT_DIR `
        --thresholds $THR `
        --topk $TOPK `
        --vit-controller $v.controller `
        --vit-aggregator max `
        --vit-aggregator-per-source $v.agg `
        --sim-json $SIM_JSON `
        --tag $v.tag
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at $($v.tag) ensemble" -ForegroundColor Red; exit 1 }

    $ENS_PARQUET = "$OUT_DIR\$($v.tag)_w1.00_ensemble.parquet"
    if (-not (Test-Path $ENS_PARQUET)) {
        Write-Host "ERROR: missing ensemble parquet at $ENS_PARQUET" -ForegroundColor Red
        exit 1
    }

    $OUT_CSV = "$OUT_DIR\$($v.tag).csv"
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
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at $($v.tag) habitat-fit" -ForegroundColor Red; exit 1 }
    Write-Host "  wrote -> $OUT_CSV"
    Write-Host ""
}

Write-Host "==========================================================="
Write-Host "Done. CSVs in $OUT_DIR :"
Write-Host "  tf_all_a030_simw_all.csv     (ALL sources alpha=0.30 + sim_w)"
Write-Host "  tf_all_a050_simw_all.csv     (ALL sources alpha=0.50 + sim_w)"
Write-Host "Compare against anchor 0.39400 priv."
Write-Host "==========================================================="
