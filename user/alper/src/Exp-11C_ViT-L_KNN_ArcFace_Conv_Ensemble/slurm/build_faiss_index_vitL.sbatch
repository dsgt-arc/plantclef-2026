#!/bin/bash
# Exp-11C Step 3: build FAISS IndexFlatIP from ViT-L LoRA+ArcFace train embeddings.
#
# Reads 1024-d 'emb1024' parquets produced by extract_embeddings_lora_vitL.py
# (which still needs to be ported from Exp-9B's extract_embeddings_lora_train.py
# — see TODO note in slurm/train_lora_vitL.sbatch tail).
#
# Reuses Exp-9B's build_faiss_index_512.py — the script is dim-agnostic and
# reads the embedding dim from the parquet column width.
#
# Usage:
#   sbatch slurm/build_faiss_index_vitL.sbatch

#SBATCH --job-name=exp11c-build-faiss-vitL
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# 1.4M x 1024 float32 = ~5.7 GB raw; allow ~96G to handle FAISS in-memory build + headroom.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
# Reuse the Exp-9B build script verbatim (it's dim-agnostic):
EXP9B_ROOT="${REPO_ROOT}/user/alper/src/Exp-9B_Improved_KNN_Retrieval"

mkdir -p logs

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"

TRAIN_EMB_DIR="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/embeddings/train"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/faiss_index"

if [ ! -d "${TRAIN_EMB_DIR}" ]; then
    echo "ERROR: ${TRAIN_EMB_DIR} missing"
    echo "Run extract_embeddings_lora_vitL.sbatch first (TODO: port from Exp-9B)."
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "=== Exp-11C: build FAISS index from ViT-L LoRA+ArcFace embeddings ==="
echo "Train embeddings (1024-d): ${TRAIN_EMB_DIR}"
echo "Output:                    ${OUT_DIR}"

python "${EXP9B_ROOT}/build_faiss_index_512.py" \
  --train-emb-dir "${TRAIN_EMB_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}" \
  --emb-col emb1024

echo ""
echo "=== Index build complete ==="
echo "Index:  ${OUT_DIR}/faiss_index_arcface_geo.bin"
echo "Labels: ${OUT_DIR}/faiss_labels_arcface_geo.npy"
