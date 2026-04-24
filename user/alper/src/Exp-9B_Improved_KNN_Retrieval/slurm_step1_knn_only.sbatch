#!/bin/bash
#SBATCH --job-name=exp9b-step1-knn
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B Step 1: Pure-kNN leaderboard comparison (beta=0.0, no softmax).
#
# Array task 0: OLD 768-d DINOv2 retriever (Exp-9 index)
# Array task 1: NEW 512-d ArcFace retriever (Exp-9B index)
#
# Both read the SAME projected test parquets (exp9b_knn/embeddings_arcface/test/)
# which preserve the original 'embedding_orig_f16' column AND carry the new
# 'emb_arcface_f16' column. Everything else is identical — tile source,
# softmax/blend config, temporal fusion, threshold — so the only variable is
# the retriever. Delta F1 between the two submissions is the pure retriever gain.
#
# Submit both CSVs. Go/no-go rule:
#   F1_new > F1_old  -> proceed to Step 2 (slurm_diff_beta_sweep.sbatch)
#   F1_new <= F1_old -> stop; the ArcFace head isn't helping kNN.

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
EXP7_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${SRC_ROOT}:${EXP7_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayTask: ${SLURM_ARRAY_TASK_ID}"

# Shared test tile parquets (projected; carry both embedding columns)
TEST_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/test"
SCALE4_DIR="${TEST_DIR}"
SCALE5_DIR="${TEST_DIR}"
SCALE6_DIR="${TEST_DIR}"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_BASE="${PLANTCLEF_ROOT}/aerten3/submissions/exp9b_step1"

# Common args across both tasks (same pipeline, same everything except retriever)
COMMON_ARGS=(
  --scale4-dir "${SCALE4_DIR}"
  --scale5-dir "${SCALE5_DIR}"
  --scale6-dir "${SCALE6_DIR}"
  --species-map "${SPECIES_MAP}"
  --class-mask-npy "${GEO_MASK}"
  --temperature 1.5
  --topk 10
  --image-topk 64
  --tile-filter-k 0
  --knn-k 20
  --knn-tau 0.07
  --betas "0.00"
  --alphas "0.15"
  --thresholds "0.100"
  --scales "4,5,6"
)

case ${SLURM_ARRAY_TASK_ID} in
  0)
    echo "=== TASK 0: OLD 768-d DINOv2 retriever (pure kNN, beta=0) ==="
    OLD_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_index_geo.bin"
    OLD_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_labels_geo.npy"
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" "${COMMON_ARGS[@]}" \
      --faiss-index "${OLD_INDEX}" \
      --faiss-labels "${OLD_LABELS}" \
      --out-dir "${OUT_BASE}/old_knn_768d" \
      --embedding-col "embedding_orig_f16" \
      --embedding-dim 768
    echo ""
    echo "Output CSV dir: ${OUT_BASE}/old_knn_768d"
    ;;
  1)
    echo "=== TASK 1: NEW 512-d ArcFace retriever (pure kNN, beta=0) ==="
    NEW_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_index_arcface_geo.bin"
    NEW_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_labels_arcface_geo.npy"
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" "${COMMON_ARGS[@]}" \
      --faiss-index "${NEW_INDEX}" \
      --faiss-labels "${NEW_LABELS}" \
      --out-dir "${OUT_BASE}/new_knn_arcface_768d" \
      --embedding-col "emb_arcface_f16" \
      --embedding-dim 768
    echo ""
    echo "Output CSV dir: ${OUT_BASE}/new_knn_arcface_768d"
    ;;
esac

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
