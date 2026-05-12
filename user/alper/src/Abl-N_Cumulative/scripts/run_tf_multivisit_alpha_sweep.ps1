# Alpha sweep across the three multi-visit sources {RNNB, CBN-PdlC, CBN-Pla}.
# Single-visit sources keep alpha=0 (no temporal fusion).
# sim_w aggregator scoped to the same three multi-visit sources.
#
# Already submitted (from previous run):
#   multivisit alpha=0.20  priv ? (just submitted)
#
# This script: alpha in {0.10, 0.30, 0.40, 0.50}.  4 new LB submissions.
#
# Reference anchors (Abl-2 level):
#   Anchor RNNB-only alpha=0.30           priv 0.39400 / pub 0.42184
#   All-source alpha=0.30 (regression)    priv 0.38173 / pub 0.38705

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

# Shared aggregator: sim_w only on the three multi-visit sources
$agg_mv = "$CFG_DIR\agg_multivisit.json"
if (-not (Test-Path $agg_mv)) {
    Set-Content -Path $agg_mv -Encoding ascii -NoNewline `
        -Value '{"RNNB":"sim_w","CBN-PdlC":"sim_w","CBN-Pla":"sim_w"}'
}

# One controller per alpha value
function Write-Controller([string]$path, [double]$a) {
    $body = '{"alpha_src": {"RNNB": ' + $a + ', "CBN-PdlC": ' + $a + ', "CBN-Pla": ' + $a + '}, "default_alpha": 0.0, "lambda_c": 0.0, "lambda_d": 0.0, "lambda_s": 0.0, "alpha_min": 0.0, "alpha_max": 0.95}'
    Set-Content -Path $path -Encoding ascii -NoNewline -Value $body
}

$alpha_values = @(0.10, 0.30, 0.40, 0.50)
$variants = @()
foreach ($a in $alpha_values) {
    $a_tag = $a.ToString("0.00").Replace(".", "p")
    $ctrl_path = "$CFG_DIR\vit_multivisit_a${a_tag}.json"
    Write-Controller $ctrl_path $a
    $variants += @{ tag = "tf_multivisit_a${a_tag}"; controller = $ctrl_path }
}

$required_base = @($NOKNN_PREFUSION, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
              $BIOGEO_NPZ, $SIM_JSON, $GEO_PROFILE, $REFINED_REF, $agg_mv)
foreach ($f in $required_base) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Multi-visit alpha sweep at Abl-2 level"
Write-Host "==========================================================="
Write-Host "Multi-visit sources: RNNB, CBN-PdlC, CBN-Pla"
Write-Host "Single-visit sources: alpha=0 (no fusion)"
Write-Host "Sweep alpha values: 0.10, 0.30, 0.40, 0.50"
Write-Host "Anchor (RNNB-only alpha=0.30): priv 0.39400 / pub 0.42184"
Write-Host ""

foreach ($v in $variants) {
    Write-Host "==========================================================="
    Write-Host "VARIANT: $($v.tag)"
    Write-Host "  controller: $($v.controller)"
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
        --vit-aggregator-per-source $agg_mv `
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
foreach ($v in $variants) { Write-Host "  $($v.tag).csv" }
Write-Host ""
Write-Host "Submit and compare to anchor 0.39400 priv + previous alpha=0.20 result."
Write-Host "==========================================================="
