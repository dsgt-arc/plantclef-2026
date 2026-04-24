#!/bin/bash
#SBATCH --job-name=exp9b-k-sweep
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B: Sweep kNN k values in parallel on the ArcFace retriever.
# Array index 0-4 maps to k=[5, 10, 20, 50, 100].
# Fixed: beta=0.80, tau=0.07, alpha=0.15, thr=0.100.
#
# Exp-9's 768-d retriever had k=20 optimal (F1=0.41153 at this config).
# ArcFace geometry may shift the optimum — re-sweep to confirm.

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
EXP7_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${SRC_ROOT}:${EXP7_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

K_VALUES=(5 10 20 50 100)
KNN_K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}
KNN_TAU=0.07

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayTask: ${SLURM_ARRAY_TASK_ID}  k=${KNN_K}"

TEST_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/test"
SCALE4_DIR="${TEST_DIR}"
SCALE5_DIR="${TEST_DIR}"
SCALE6_DIR="${TEST_DIR}"

FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_index_arcface_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_labels_arcface_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp9b_knn_k_sweep"

python "${SRC_ROOT}/multiscale_aggregate_knn.py" \
  --scale4-dir "${SCALE4_DIR}" \
  --scale5-dir "${SCALE5_DIR}" \
  --scale6-dir "${SCALE6_DIR}" \
  --faiss-index "${FAISS_INDEX}" \
  --faiss-labels "${FAISS_LABELS}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}" \
  --temperature 1.5 \
  --topk 10 \
  --image-topk 64 \
  --tile-filter-k 0 \
  --knn-k ${KNN_K} \
  --knn-tau ${KNN_TAU} \
  --betas "0.80" \
  --alphas "0.15" \
  --thresholds "0.100" \
  --scales "4,5,6" \
  --knn-ks "${KNN_K}" \
  --knn-taus "${KNN_TAU}" \
  --embedding-col "emb_arcface_f16" \
  --embedding-dim 512

echo "=== Done: k=${KNN_K}, tau=${KNN_TAU} ==="
