# Light tuning of TEMPORAL FUSION application at the Abl-2 level.
#
# Anchor (Abl-2, vit_rnnb030.json + RNNB:sim_w):  priv 0.39400 / pub 0.42184
# Abl-3 (no temporal fusion at all):              priv 0.38961 / pub 0.41178
#
# Sweep dimensions:
#   Intensity (RNNB only):
#     vit_rnnb015.json  -- RNNB alpha=0.15 (lighter)
#     vit_rnnb050.json  -- RNNB alpha=0.50 (heavier)
#     vit_rnnb070.json  -- RNNB alpha=0.70 (heaviest)
#     aggregator-per-source stays at {"RNNB":"sim_w"} for these
#
#   Breadth (also apply temporal fusion to OPTMix):
#     vit_RNNB_OPTMix.json   -- RNNB + OPTMix both at alpha=0.30
#     aggregator-per-source extended to {"RNNB":"sim_w","OPTMix":"sim_w"}
#
# Each variant: re-fuse ensemble step on the no-kNN prefusion, then
# apply_habitat_fit.py with beta=1.0 + skip masks (Abl-2 cumulative config).
#
# Usage:  pwsh .\run_temporal_fusion_sweep.ps1

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

$OUT_DIR = "$ABLN\outputs\tf_sweep"
$AGG_DIR = "$OUT_DIR\agg_jsons"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null
New-Item -Force -ItemType Directory $AGG_DIR | Out-Null

# Write each variant's aggregator JSON to a file (PowerShell can't reliably
# pass {"x":"y"} on the command line without double-quote stripping).
$agg_rnnb_only   = "$AGG_DIR\agg_rnnb_only.json"
$agg_rnnb_optmix = "$AGG_DIR\agg_rnnb_optmix.json"
Set-Content -Path $agg_rnnb_only   -Value '{"RNNB":"sim_w"}'                   -Encoding ascii -NoNewline
Set-Content -Path $agg_rnnb_optmix -Value '{"RNNB":"sim_w","OPTMix":"sim_w"}'  -Encoding ascii -NoNewline

$variants = @(
    @{ tag = "tf_rnnb015";     controller = "$EXP14\stage_configs\controllers\vit_rnnb015.json";     agg = $agg_rnnb_only },
    @{ tag = "tf_rnnb050";     controller = "$EXP14\stage_configs\controllers\vit_rnnb050.json";     agg = $agg_rnnb_only },
    @{ tag = "tf_rnnb070";     controller = "$EXP14\stage_configs\controllers\vit_rnnb070.json";     agg = $agg_rnnb_only },
    @{ tag = "tf_rnnb_optmix"; controller = "$EXP14\stage_configs\controllers\vit_RNNB_OPTMix.json"; agg = $agg_rnnb_optmix }
)

$required = @($NOKNN_PREFUSION, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $SIM_JSON, $GEO_PROFILE, $REFINED_REF)
foreach ($v in $variants) { $required += $v.controller; $required += $v.agg }
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Temporal fusion sweep at Abl-2 level"
Write-Host "==========================================================="
Write-Host "Anchor: vit_rnnb030.json + RNNB:sim_w  -> priv 0.39400 / pub 0.42184"
Write-Host "Floor : no temporal fusion             -> priv 0.38961 / pub 0.41178"
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
Write-Host "Done. 4 CSVs in $OUT_DIR :"
Write-Host "  tf_rnnb015.csv       (RNNB alpha=0.15)"
Write-Host "  tf_rnnb050.csv       (RNNB alpha=0.50)"
Write-Host "  tf_rnnb070.csv       (RNNB alpha=0.70)"
Write-Host "  tf_rnnb_optmix.csv   (RNNB + OPTMix both at 0.30)"
Write-Host ""
Write-Host "Compare each against anchor 0.39400 priv."
Write-Host "==========================================================="
