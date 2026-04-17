#!/bin/bash
#SBATCH --job-name=exp9-build-faiss
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

# Exp-9 Phase 1: Build FAISS index from pre-extracted training embeddings.
#
# Loads ~1.4M CLS embeddings, filters to ~860K by geo mask (~4800 species),
# L2-normalizes, and builds IndexFlatIP for cosine similarity search.
#
# CPU-only job. Expected: ~30 min, ~10 GB RAM peak.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9_kNN_Retrieval_Ensemble"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

TRAIN_EMB_DIR="${PLANTCLEF_ROOT}/embeddings/train"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index"

echo "=== Exp-9: Build FAISS Index ==="
echo "Training embeddings: ${TRAIN_EMB_DIR}"
echo "Output: ${OUT_DIR}"

python "${SRC_ROOT}/build_faiss_index.py" \
  --train-emb-dir "${TRAIN_EMB_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}"

echo ""
echo "=== Index build complete ==="
echo "Index: ${OUT_DIR}/faiss_index_geo.bin"
echo "Labels: ${OUT_DIR}/faiss_labels_geo.npy"
