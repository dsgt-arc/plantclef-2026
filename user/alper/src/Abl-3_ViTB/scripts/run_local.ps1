# Abl-3 local runner: ViT-B vs ViT-L backbone ablation (no kNN, no ConvNeXt).
#
# Mirrors Abl-2's pipeline exactly — same downstream stack, only the backbone
# tile parquets change. Reuses existing Python scripts (all backbone-agnostic).
#
# Comparison anchors (Abl-2 with ViT-L):
#   bare CSV    : private 0.38702 / public 0.42181
#   full stack  : private 0.44053 / public 0.44953
#
# Usage:
#   pwsh .\Abl-3_ViTB\scripts\run_local.ps1
#
# Outputs (placed under Abl-3_ViTB/outputs/):
#   abl3_vitB_noknn_prefusion.parquet                  (no-kNN ViT-B prefusion)
#   abl3_vitB_noknn_w1.00_ensemble.parquet             (ViT-only ensemble, no kNN, no ConvNeXt)
#   abl3_vitB_noknn_w1.00_thr0.085_top10.csv           (bare ViT-B, no habitat-fit) → vs Abl-2 bare 0.38702
#   abl3_s9_B_R250_vitB.csv                            (full Abl-3 stack)            → vs Abl-2 full 0.44053

$ErrorActionPreference = "Stop"

# ---- B_R250-locked parameters (held identical to Abl-2 for apples-to-apples) ----
$THR = "0.085"
$TOPK = "10"
$TEMPERATURE = "1.5"
$IMAGE_TOPK = "64"
$HAB_BETA = "0.10"
$HAB_R = "250"
$HAB_N = "40"

# ---- Roots ----
$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABL1  = "$REPO\user\alper\src\Abl-1_CNext"
$ABL2  = "$REPO\user\alper\src\Abl-2_KNN"
$ABL3  = "$REPO\user\alper\src\Abl-3_ViTB"

$PACE_DL  = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"
$VITB_DIR = "$PACE_DL\vitB_multiscale\exp7_multiscale"

# ---- Inputs (matched to Abl-2 except for the backbone-tile parquets) ----
$VITB_SCALES = @(
    "$VITB_DIR\scale_4x4.parquet",
    "$VITB_DIR\scale_5x5.parquet",
    "$VITB_DIR\scale_6x6.parquet",
    "$VITB_DIR\scale_7x7.parquet"
)
$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$VIT_CONTROLLER= "$EXP14\stage_configs\controllers\vit_rnnb030.json"
$SIM_JSON      = "$EXP14\outputs\signals\visit_similarity.json"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

# ---- Outputs ----
$OUT_DIR     = "$ABL3\outputs"
$PREF_DIR    = "$OUT_DIR\prefusion"
$PS_CFG_DIR  = "$OUT_DIR\per_src_configs"
New-Item -Force -ItemType Directory $OUT_DIR     | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR    | Out-Null
New-Item -Force -ItemType Directory $PS_CFG_DIR  | Out-Null

# B_R250-locked per-source admission
$LOCKED_PS = "$PS_CFG_DIR\locked.json"
'{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}' | Set-Content -NoNewline -Path $LOCKED_PS

# ---- Pre-flight sanity ----
$required = @($SPECIES_MAP, $GEO_MASK, $HABITAT_MASK, $BIOGEO_NPZ, $VIT_CONTROLLER,
              $SIM_JSON, $GEO_PROFILE, $REFINED_REF) + $VITB_SCALES
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

$TAG = "abl3_vitB_noknn"

Write-Host "=== Abl-3 ViT-B backbone ablation (no kNN, no ConvNeXt) ==="
Write-Host "Reference scores (Abl-2 with ViT-L):"
Write-Host "  bare ViT-only  priv 0.38702 / pub 0.42181"
Write-Host "  full stack     priv 0.44053 / pub 0.44953"
Write-Host "Habitat-fit lock: beta=$HAB_BETA, R=$HAB_R, n=$HAB_N, alt=on, thr=$THR, k=$TOPK"
Write-Host "Per-source admission: OPTMix n_floor=2, LISAH n_floor=2"
Write-Host ""

# ------------------------------------------------------------
# Step A: no-kNN pre-fusion (classifier softmax only, ViT-B tiles)
# ------------------------------------------------------------
Write-Host "--- Step A: no-kNN pre-fusion from ViT-B tiles ---"
python "$ABL2\scripts\make_vitL_prefusion_no_knn.py" `
    --vit-scales $VITB_SCALES `
    --class-mask-npy $GEO_MASK `
    --out-dir $PREF_DIR `
    --temperature $TEMPERATURE `
    --image-topk $IMAGE_TOPK `
    --tag $TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step A" -ForegroundColor Red; exit 1 }

$PREFUSION = "$PREF_DIR\${TAG}_prefusion.parquet"
if (-not (Test-Path $PREFUSION)) {
    Write-Host "ERROR: prefusion parquet missing at $PREFUSION" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------
# Step B: ViT-only ensemble (re-uses Abl-1's script — no ConvNeXt branch)
#         Same vit_rnnb030.json controller and sim_w-on-RNNB as Abl-1/Abl-2.
# ------------------------------------------------------------
Write-Host ""
Write-Host "--- Step B: ViT-only ensemble (controller + sim_w-on-RNNB) ---"
python "$ABL1\scripts\ensemble_vit_only.py" `
    --vit-prefusion-parquet $PREFUSION `
    --species-map $SPECIES_MAP `
    --class-mask-npy $GEO_MASK `
    --out-dir $OUT_DIR `
    --thresholds $THR `
    --topk $TOPK `
    --vit-controller $VIT_CONTROLLER `
    --vit-aggregator max `
    --vit-aggregator-per-source '{"RNNB":"sim_w"}' `
    --sim-json $SIM_JSON `
    --tag $TAG
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step B" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"
if (-not (Test-Path $ENS_PARQUET)) {
    Write-Host "ERROR: ensemble parquet missing at $ENS_PARQUET" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------
# Step C: habitat-fit + per-source admission (B_R250 lock)
# ------------------------------------------------------------
Write-Host ""
Write-Host "--- Step C: habitat-fit (beta=$HAB_BETA, R=$HAB_R, n=$HAB_N) + admission ---"
$OUT_CSV = "$OUT_DIR\abl3_s9_B_R250_vitB.csv"
python "$EXP16\scripts\apply_habitat_fit.py" `
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
    --per-source-json $LOCKED_PS `
    --out-csv $OUT_CSV
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step C" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=== Done. CSVs in $OUT_DIR ==="
Write-Host "  bare ViT-B (no habitat-fit):  ${TAG}_w1.00_thr${THR}_top${TOPK}.csv  --> vs Abl-2 bare 0.38702 priv"
Write-Host "  full Abl-3 stack:              abl3_s9_B_R250_vitB.csv             --> vs Abl-2 full 0.44053 priv"
