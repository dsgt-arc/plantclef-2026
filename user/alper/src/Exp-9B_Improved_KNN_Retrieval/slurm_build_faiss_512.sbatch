#!/bin/bash
#SBATCH --job-name=exp9b-build-faiss-lora
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B: build a FAISS IndexFlatIP from Tier 2 LoRA-projected train embeddings.
#
# Reads the 768-d 'emb768' parquets produced by extract_embeddings_lora_train.py
# (output of slurm_extract_lora_train.sbatch). Writes the index files to
# the same `faiss_index/` directory the Exp-9B inference scripts already
# reference, so Step 1 / Step 2 sbatches work unchanged.
#
# Note: the script name retains "_512" for historical reasons but is fully
# dim-agnostic — it reads the dim from the parquet column width and builds
# IndexFlatIP at that dim.
#
# CPU-only job. 1.4M x 768 float32 = ~4.3 GB to load -> larger memory budget.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"

TRAIN_EMB_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface_lora/train"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index"

echo "=== Exp-9B: build FAISS index from Tier 2 LoRA embeddings ==="
echo "Train embeddings (768-d): ${TRAIN_EMB_DIR}"
echo "Output:                   ${OUT_DIR}"

python "${SRC_ROOT}/build_faiss_index_512.py" \
  --train-emb-dir "${TRAIN_EMB_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}" \
  --emb-col emb768

echo ""
echo "=== Index build complete ==="
echo "Index:  ${OUT_DIR}/faiss_index_arcface_geo.bin"
echo "Labels: ${OUT_DIR}/faiss_labels_arcface_geo.npy"
