# Two more multi-scale transfer tests at the FULL Baseline downstream:
#   3+4+5    (50 tiles/image)
#   3+4+5+6  (86 tiles/image)
#
# Baseline anchor (4+5+6+7): priv 0.45732 / pub 0.47351
# Bare-level results:
#   3+4+5    bare priv 0.39865 / pub 0.38202
#   3+4+5+6  bare priv 0.40079 / pub 0.39835
#   4+5+6+7  bare priv 0.38961 / pub 0.41178
#
# Hypothesis (per the 3+6 transfer null result):
#   Bare-level advantages of scale-3-containing combos likely do not transfer
#   to the full Baseline; downstream stack equalizes.

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14C = "$REPO\user\alper\src\Exp-14C_ViT-L_LoRA_Ensemble"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"
$ARCFACE_DIR = "$PACE_DL\exp11c_vitL_lora_arcface\embeddings\test"
$FAISS_DIR   = "$PACE_DL\exp11c_vitL_lora_arcface\faiss_index"

$FAISS_INDEX   = "$FAISS_DIR\faiss_index_arcface_geo.bin"
$FAISS_LABELS  = "$FAISS_DIR\faiss_labels_arcface_geo.npy"

$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$VIT_CONTROLLER= "$EXP14\stage_configs\controllers\vit_rnnb030.json"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$THR = "0.085"; $TOPK = "10"
$HAB_BETA = "0.10"; $HAB_R = "250"; $HAB_N = "40"
$BETA_MULTI = "0.70"
$BETA_SINGLE = "1.00"

$OUT_DIR = "$ABLN\outputs\scale_more_baseline_knn"
$PREF_DIR = "$OUT_DIR\prefusion"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR  | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR  | Out-Null

$agg = "$CFG_DIR\agg_rnnb_only.json"
Set-Content -Path $agg -Encoding ascii -NoNewline -Value '{"RNNB":"sim_w"}'

function Scales([int[]]$scales) {
    return $scales | ForEach-Object { "$ARCFACE_DIR\scale_${_}x${_}.parquet" }
}

$variants = @(
    @{ tag = "scale345"; scales = @(3,4,5);   pref_tag = "vitL_scale345" },
    @{ tag = "scale3456"; scales = @(3,4,5,6); pref_tag = "vitL_scale3456" }
)

# Pre-flight
foreach ($v in $variants) {
    foreach ($s in $v.scales) {
        $p = "$ARCFACE_DIR\scale_${s}x${s}.parquet"
        if (-not (Test-Path $p)) {
            Write-Host "ERROR: missing tile parquet: $p" -ForegroundColor Red
            exit 1
        }
    }
}
foreach ($f in @($FAISS_INDEX, $FAISS_LABELS, $SPECIES_MAP, $GEO_MASK,
                  $HABITAT_MASK, $BIOGEO_NPZ, $VIT_CONTROLLER, $SIM_JSON,
                  $GEO_PROFILE, $REFINED_REF, $agg)) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file: $f" -ForegroundColor Red
        exit 1
    }
}

foreach ($v in $variants) {
    $TAG = "$($v.tag)_baseline_knn"
    Write-Host "==========================================================="
    Write-Host "VARIANT: $TAG  (scales: $($v.scales -join ','))"
    Write-Host "==========================================================="

    $scale_paths = Scales $v.scales

    # Step A: kNN prefusion
    & $PYTHON "$EXP14C\make_vitL_prefusion.py" `
        --vit-scales $scale_paths `
        --faiss-index $FAISS_INDEX `
        --faiss-labels $FAISS_LABELS `
        --species-map $SPECIES_MAP `
        --class-mask-npy $GEO_MASK `
        --out-dir $PREF_DIR `
        --temperature 1.5 `
        --knn-k 20 `
        --knn-tau 0.07 `
        --beta-multis $BETA_MULTI `
        --beta-single $BETA_SINGLE `
        --image-topk 64 `
        --tag $v.pref_tag
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED Step A ($TAG)" -ForegroundColor Red; exit 1 }

    $PREFUSION = "$PREF_DIR\$($v.pref_tag)_prefusion_b0.70.parquet"

    # Step B: ensemble (Baseline temporal fusion)
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
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED Step B ($TAG)" -ForegroundColor Red; exit 1 }

    $ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"

    # Step C: habitat-fit at Baseline (no admission)
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
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED Step C ($TAG)" -ForegroundColor Red; exit 1 }
    Write-Host "  wrote -> $OUT_CSV"
    Write-Host ""
}

Write-Host "==========================================================="
Write-Host "Done. Compare to Baseline (4+5+6+7): 0.45732 priv / 0.47351 pub"
Write-Host "==========================================================="
