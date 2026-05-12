# Test scales 3+6 at the FULL Baseline downstream (kNN ON).
#
# Inputs (all local now):
#   ViT-L per-tile parquets WITH emb_arcface_f16:
#     - PACE_downloads/exp11c_vitL_lora_arcface/embeddings/test/scale_3x3.parquet
#     - PACE_downloads/exp11c_vitL_lora_arcface/embeddings/test/scale_6x6.parquet
#   FAISS index over training-set ArcFace embeddings:
#     - PACE_downloads/exp11c_vitL_lora_arcface/faiss_index/faiss_index_arcface_geo.bin
#     - PACE_downloads/exp11c_vitL_lora_arcface/faiss_index/faiss_labels_arcface_geo.npy
#
# Pipeline:
#   Step A: make_vitL_prefusion.py with kNN at beta_multi=0.70 (Baseline lock)
#   Step B: ensemble_vit_only.py with vit_rnnb030 controller + RNNB sim_w
#   Step C: apply_habitat_fit.py at Baseline (beta=0.10, R=250, n=40,
#           masks ON, no admission overrides)
#
# Reference:
#   Baseline (4+5+6+7 with kNN + full downstream): priv 0.45732 / pub 0.47351
#   Abl-1 (4+5+6+7 minus kNN, full downstream):    priv 0.44785 / pub 0.45168
#   Abl-3 anchor (4+5+6+7 bare):                    priv 0.38961 / pub 0.41178
#   Abl-4 3+6 (bare):                                priv 0.40637 / pub 0.39639
#
# Predicted: this run (3+6 + full Baseline downstream WITH kNN)
#   Optimistic projection: 0.45732 + 0.01676 = ~0.47408 priv
#   Naive projection from "Abl-1 + 3+6 lift + kNN":
#     0.44785 + 0.01676 + 0.00947 = ~0.47408 priv
#
# Usage:  pwsh .\run_scale36_at_baseline.ps1

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

# Inputs
$VIT_SCALE_3   = "$ARCFACE_DIR\scale_3x3.parquet"
$VIT_SCALE_6   = "$ARCFACE_DIR\scale_6x6.parquet"
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

# Baseline-locked params
$THR = "0.085"; $TOPK = "10"
$HAB_BETA = "0.10"; $HAB_R = "250"; $HAB_N = "40"
$BETA_MULTI = "0.70"   # Baseline kNN blend
$BETA_SINGLE = "1.00"  # single-visit: no kNN

$OUT_DIR = "$ABLN\outputs\scale36_baseline_knn"
$PREF_DIR = "$OUT_DIR\prefusion"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR  | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR  | Out-Null

# Aggregator JSON file (PowerShell can't pass JSON literal safely)
$agg = "$CFG_DIR\agg_rnnb_only.json"
Set-Content -Path $agg -Encoding ascii -NoNewline -Value '{"RNNB":"sim_w"}'

$required = @($VIT_SCALE_3, $VIT_SCALE_6, $FAISS_INDEX, $FAISS_LABELS,
              $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK, $BIOGEO_NPZ,
              $VIT_CONTROLLER, $SIM_JSON, $GEO_PROFILE, $REFINED_REF, $agg)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file: $f" -ForegroundColor Red
        exit 1
    }
}

$TAG = "scale36_baseline_knn"

Write-Host "==========================================================="
Write-Host "Scale 3+6 at full Baseline (kNN ON)"
Write-Host "==========================================================="
Write-Host "Anchor (4+5+6+7 + kNN + full downstream): priv 0.45732 / pub 0.47351"
Write-Host "Targeting: ~0.474 priv if the 3+6 multi-scale lift transfers."
Write-Host ""

# ---------------------------------------------------------------
# Step A: build kNN-enabled prefusion from local ArcFace parquets
# ---------------------------------------------------------------
Write-Host "--- Step A: kNN prefusion build (scales 3+6, beta_multi=0.70) ---"
& $PYTHON "$EXP14C\make_vitL_prefusion.py" `
    --vit-scales $VIT_SCALE_3 $VIT_SCALE_6 `
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
    --tag "vitL_scale36"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step A (kNN prefusion)" -ForegroundColor Red; exit 1 }

$PREFUSION = "$PREF_DIR\vitL_scale36_prefusion_b0.70.parquet"
if (-not (Test-Path $PREFUSION)) {
    Write-Host "ERROR: prefusion missing at $PREFUSION" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------
# Step B: ensemble with temporal fusion ON (Baseline)
# ---------------------------------------------------------------
Write-Host ""
Write-Host "--- Step B: ensemble (vit_rnnb030 + RNNB sim_w) ---"
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
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step B (ensemble)" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"

# ---------------------------------------------------------------
# Step C: habitat-fit at Baseline (beta=0.10, masks ON, no admission)
# ---------------------------------------------------------------
Write-Host ""
Write-Host "--- Step C: habitat-fit (Baseline lock) ---"
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
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step C (habitat-fit)" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "==========================================================="
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Compare to Baseline 0.45732 priv / 0.47351 pub"
Write-Host "==========================================================="
