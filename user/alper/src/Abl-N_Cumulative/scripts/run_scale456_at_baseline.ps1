# Test scales 4+5+6 at the FULL Baseline downstream (kNN ON).
#
# Anchor (production 4+5+6+7 at Baseline): priv 0.45732 / pub 0.47351
# Bare-level result for 4+5+6:              priv 0.39317 / pub 0.39882
# Bare-level result for 4+5+6+7:            priv 0.38961 / pub 0.41178
#   -> 4+5+6 beat 4+5+6+7 by +0.00356 priv at bare level. Does it transfer?

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

$VIT_SCALE_4   = "$ARCFACE_DIR\scale_4x4.parquet"
$VIT_SCALE_5   = "$ARCFACE_DIR\scale_5x5.parquet"
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

$THR = "0.085"; $TOPK = "10"
$HAB_BETA = "0.10"; $HAB_R = "250"; $HAB_N = "40"
$BETA_MULTI = "0.70"
$BETA_SINGLE = "1.00"

$OUT_DIR = "$ABLN\outputs\scale456_baseline_knn"
$PREF_DIR = "$OUT_DIR\prefusion"
$CFG_DIR = "$OUT_DIR\configs"
New-Item -Force -ItemType Directory $OUT_DIR  | Out-Null
New-Item -Force -ItemType Directory $PREF_DIR | Out-Null
New-Item -Force -ItemType Directory $CFG_DIR  | Out-Null

$agg = "$CFG_DIR\agg_rnnb_only.json"
Set-Content -Path $agg -Encoding ascii -NoNewline -Value '{"RNNB":"sim_w"}'

$required = @($VIT_SCALE_4, $VIT_SCALE_5, $VIT_SCALE_6, $FAISS_INDEX, $FAISS_LABELS,
              $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK, $BIOGEO_NPZ,
              $VIT_CONTROLLER, $SIM_JSON, $GEO_PROFILE, $REFINED_REF, $agg)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing file: $f" -ForegroundColor Red
        exit 1
    }
}

$TAG = "scale456_baseline_knn"

Write-Host "==========================================================="
Write-Host "Scale 4+5+6 at full Baseline (kNN ON)"
Write-Host "==========================================================="

# Step A: kNN-enabled prefusion at scales 4+5+6
Write-Host "--- Step A: kNN prefusion (4+5+6, beta_multi=0.70) ---"
& $PYTHON "$EXP14C\make_vitL_prefusion.py" `
    --vit-scales $VIT_SCALE_4 $VIT_SCALE_5 $VIT_SCALE_6 `
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
    --tag "vitL_scale456"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step A" -ForegroundColor Red; exit 1 }

$PREFUSION = "$PREF_DIR\vitL_scale456_prefusion_b0.70.parquet"

# Step B: ensemble (Baseline temporal fusion)
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
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step B" -ForegroundColor Red; exit 1 }

$ENS_PARQUET = "$OUT_DIR\${TAG}_w1.00_ensemble.parquet"

# Step C: habitat-fit at Baseline (no admission)
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
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Step C" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Done. CSV: $OUT_CSV"
Write-Host "Compare to Baseline (4+5+6+7) 0.45732 priv / 0.47351 pub"
