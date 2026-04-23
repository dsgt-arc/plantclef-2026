#!/bin/bash
#SBATCH --job-name=exp9b-build-faiss-512
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B: build a 512-d FAISS IndexFlatIP from ArcFace-projected train embeddings.
# Mirrors Exp-9 build_faiss_index.sbatch but reads the projected parquets
# and writes to exp9b_knn/faiss_index.
#
# CPU-only job. Expected: ~20-30 min, ~10 GB RAM peak.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"

TRAIN_EMB_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/train"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index"

echo "=== Exp-9B: build 512-d FAISS index ==="
echo "Train embeddings (512-d): ${TRAIN_EMB_DIR}"
echo "Output:                   ${OUT_DIR}"

python "${SRC_ROOT}/build_faiss_index_512.py" \
  --train-emb-dir "${TRAIN_EMB_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}" \
  --emb-col emb512

echo ""
echo "=== Index build complete ==="
echo "Index:  ${OUT_DIR}/faiss_index_arcface_geo.bin"
echo "Labels: ${OUT_DIR}/faiss_labels_arcface_geo.npy"
