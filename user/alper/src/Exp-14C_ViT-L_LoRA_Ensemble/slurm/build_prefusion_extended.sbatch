#!/bin/bash
# Exp-14C Stage 1ext-a: build pre-fusion parquets at LOWER beta_multi values
# {0.55, 0.60, 0.65, 0.70}. Stage 1 showed lower beta wins monotonically;
# this extends the search downward.
#
# Re-runs the FAISS query pass (~1.5–2h) — the precompute step doesn't get
# cached across sbatch invocations, so this is the dominant cost.
#
# Usage:
#   sbatch slurm/build_prefusion_extended.sbatch

#SBATCH --job-name=exp14c-prefusion-ext
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-14C_ViT-L_LoRA_Ensemble"

mkdir -p logs

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

VIT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/embeddings/test"
FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/faiss_index/faiss_index_arcface_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/faiss_index/faiss_labels_arcface_geo.npy"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"

mkdir -p "${OUT_DIR}"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C Stage 1ext-a: extend beta down to {0.55, 0.60, 0.65, 0.70} ==="

python "${SRC_ROOT}/make_vitL_prefusion.py" \
    --vit-scales \
        "${VIT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${VIT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${VIT_PARQUET_ROOT}/scale_6x6.parquet" \
        "${VIT_PARQUET_ROOT}/scale_7x7.parquet" \
    --faiss-index "${FAISS_INDEX}" \
    --faiss-labels "${FAISS_LABELS}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --knn-k 20 \
    --knn-tau 0.07 \
    --beta-multis "0.55,0.60,0.65,0.70" \
    --beta-single 1.00 \
    --image-topk 64 \
    --tag "vitL"

echo ""
echo "=== Stage 1ext-a done ==="
echo "Should now have prefusion parquets at b={0.55,0.60,0.65,0.70,0.75,0.85,0.95}:"
ls -lh "${OUT_DIR}/vitL_prefusion_b"*.parquet 2>/dev/null || true
echo ""
echo "Next: sbatch slurm/ensemble_sweep_stage1ext.sbatch"
