#!/bin/bash
# Abl-2 — kNN knockout ablation (cumulative from Abl-1).
#
# Cumulative path:
#   B_R250 (full)        → private 0.44649, public 0.48414
#   Abl-1 (− ConvNeXt)   → private 0.45001, public 0.47136
#   Abl-2 (− ConvNeXt − kNN)  ← this run
#
# Pipeline (every other layer held at B_R250 lock):
#   - ViT-L pre-fusion: pure classifier softmax, NO kNN  (beta_multi → 1.0)
#   - ViT-only ensemble (reuses Abl-1's ensemble_vit_only.py)
#     - vit_rnnb030.json controller (α=0.30 RNNB temporal fusion)
#     - sim_w-on-RNNB per-source aggregator
#   - apply_habitat_fit.py at B_R250 lock:
#     β=0.10, R=250, n_near=40, alt-on, thr=0.085, k=10
#     per-source {OPTMix: n_floor=2, LISAH: n_floor=2}
#
# Methodology note (fixed-knob ablation): the kNN β_match=0.70 was tuned
# with kNN present; the habitat-fit knobs were tuned on top of the
# kNN-included ensemble. They are NOT re-tuned here. Reported ΔF1 is a
# lower bound on the true marginal cost of removing kNN.
#
# Per memory (Exp-16 veg-fraction findings): kNN was estimated to
# contribute ~0.03 F1 on public LB. Private contribution unknown until
# this ablation runs.
#
# Output:
#   abl2_vitL_noknn_prefusion.parquet           (no-kNN ViT pre-fusion)
#   abl2_vitL_noknn_w1.00_ensemble.parquet      (ViT-only ensemble, no kNN, no ConvNeXt)
#   abl2_vitL_noknn_w1.00_thr0.085_top10.csv    (bare CSV, no habitat-fit)
#   abl2_s9_B_R250_noknn.csv                    ← HEADLINE: B_R250 stack minus ConvNeXt minus kNN
#
# Usage:
#   sbatch slurm/ablation_no_knn.sbatch

#SBATCH --job-name=abl2-no-knn
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# B_R250-locked parameters (do not change without a separate re-tune ablation)
THR="${THR:-0.085}"
TOPK="${TOPK:-10}"
TEMPERATURE="${TEMPERATURE:-1.5}"
IMAGE_TOPK="${IMAGE_TOPK:-64}"
HAB_BETA="${HAB_BETA:-0.10}"
HAB_R="${HAB_R:-250}"
HAB_N="${HAB_N:-40}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
EXP16_ROOT="${REPO_ROOT}/user/alper/src/Exp-16_Multi_Species_Discovery"
ABL1_ROOT="${REPO_ROOT}/user/alper/src/Abl-1_CNext"
ABL2_ROOT="${REPO_ROOT}/user/alper/src/Abl-2_KNN"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

# Inputs (per-tile ViT-L parquets — same source as Exp-14C build_prefusion_extended.sbatch)
VIT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/embeddings/test"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
VIT_CONTROLLER="${EXP14_ROOT}/stage_configs/controllers/vit_rnnb030.json"
SIM_JSON="${EXP14_ROOT}/outputs/signals/visit_similarity.json"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/abl2_no_knn"
PS_DIR="${OUT_DIR}/per_src_configs"
PREFUSION_DIR="${OUT_DIR}/prefusion"
mkdir -p "${OUT_DIR}" "${PS_DIR}" "${PREFUSION_DIR}"

# Locked per-source admission from Stage 9 B_R250
cat > "${PS_DIR}/locked.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
LOCKED_PS="${PS_DIR}/locked.json"

# Pre-flight sanity (no FAISS files required for this ablation)
for f in "${VIT_PARQUET_ROOT}/scale_4x4.parquet" \
         "${VIT_PARQUET_ROOT}/scale_5x5.parquet" \
         "${VIT_PARQUET_ROOT}/scale_6x6.parquet" \
         "${VIT_PARQUET_ROOT}/scale_7x7.parquet" \
         "${HABITAT_MASK}" "${BIOGEO_NPZ}" \
         "${VIT_CONTROLLER}" "${SIM_JSON}" \
         "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing required file: $f"; exit 1; fi
done

TAG="abl2_vitL_noknn"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Abl-2 ConvNeXt + kNN knockout (cumulative from Abl-1) ==="
echo "Reference scores:"
echo "  B_R250 (full)        priv 0.44649 / pub 0.48414"
echo "  Abl-1 (− ConvNeXt)   priv 0.45001 / pub 0.47136"
echo "Habitat-fit: β=${HAB_BETA}, R=${HAB_R}, n=${HAB_N}, alt=on, thr=${THR}, k=${TOPK}"
echo "Per-source admission: OPTMix n_floor=2, LISAH n_floor=2"

# ---------------------------------------------------------------
# Step A: build no-kNN pre-fusion parquet (pure classifier softmax)
# ---------------------------------------------------------------
echo ""
echo "--- Step A: no-kNN pre-fusion (classifier-only) ---"
python "${ABL2_ROOT}/scripts/make_vitL_prefusion_no_knn.py" \
    --vit-scales \
        "${VIT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${VIT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${VIT_PARQUET_ROOT}/scale_6x6.parquet" \
        "${VIT_PARQUET_ROOT}/scale_7x7.parquet" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${PREFUSION_DIR}" \
    --temperature "${TEMPERATURE}" \
    --image-topk "${IMAGE_TOPK}" \
    --tag "${TAG}"

PREFUSION="${PREFUSION_DIR}/${TAG}_prefusion.parquet"
if [ ! -f "${PREFUSION}" ]; then
    echo "ERROR: pre-fusion parquet not saved at ${PREFUSION}"
    exit 1
fi

# ---------------------------------------------------------------
# Step B: ViT-only ensemble (reuse Abl-1's script — no ConvNeXt branch)
# ---------------------------------------------------------------
echo ""
echo "--- Step B: ViT-only fusion + ensemble parquet ---"
python "${ABL1_ROOT}/scripts/ensemble_vit_only.py" \
    --vit-prefusion-parquet "${PREFUSION}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --thresholds "${THR}" \
    --topk "${TOPK}" \
    --vit-controller "${VIT_CONTROLLER}" \
    --vit-aggregator max \
    --vit-aggregator-per-source '{"RNNB":"sim_w"}' \
    --sim-json "${SIM_JSON}" \
    --tag "${TAG}"

ENS_PARQUET="${OUT_DIR}/${TAG}_w1.00_ensemble.parquet"
if [ ! -f "${ENS_PARQUET}" ]; then
    echo "ERROR: ensemble parquet not saved at ${ENS_PARQUET}"
    exit 1
fi

# ---------------------------------------------------------------
# Step C: habitat-fit + per-source admission (B_R250 lock)
# ---------------------------------------------------------------
echo ""
echo "--- Step C: habitat-fit (β=${HAB_BETA}, R=${HAB_R}, n=${HAB_N}) + admission ---"
OUT_CSV="${OUT_DIR}/abl2_s9_B_R250_noknn.csv"
python "${EXP16_ROOT}/scripts/apply_habitat_fit.py" \
    --ensemble-parquet "${ENS_PARQUET}" \
    --geo-profile "${GEO_PROFILE}" \
    --species-map "${SPECIES_MAP}" \
    --geo-mask "${GEO_MASK}" \
    --habitat-mask "${HABITAT_MASK}" \
    --biogeo-npz "${BIOGEO_NPZ}" \
    --refined-ref-json "${REFINED_REF}" \
    --radius-km "${HAB_R}" \
    --beta "${HAB_BETA}" \
    --n-near-min "${HAB_N}" \
    --use-altitude \
    --threshold "${THR}" \
    --topk "${TOPK}" \
    --n-floor 3 \
    --ratio-floor 0.5 \
    --per-source-json "${LOCKED_PS}" \
    --out-csv "${OUT_CSV}"

echo ""
echo "=== Done. CSVs in ${OUT_DIR} ==="
echo "  bare ViT-only no-kNN (no habitat-fit): ${TAG}_w1.00_thr${THR}_top${TOPK}.csv"
echo "  full Abl-2 stack:                       abl2_s9_B_R250_noknn.csv"
echo ""
echo "LB submissions to record:"
echo "  abl2_s9_B_R250_noknn.csv -> compare private vs Abl-1 (0.45001) and B_R250 (0.44649)"
