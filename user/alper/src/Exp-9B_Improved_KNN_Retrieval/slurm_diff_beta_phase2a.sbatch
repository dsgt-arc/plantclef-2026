#!/bin/bash
#SBATCH --job-name=exp9b-diff-beta-2a
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

# Phase 2a: confirm beta_multi=0.80 is the true peak by testing higher values.
# Initial sweep tested beta_multi in {0.65, 0.70, 0.75, 0.80} with peak F1=0.41976
# at beta_multi=0.80. This adds {0.85, 0.90} above that to verify the optimum
# isn't actually at higher beta (less kNN). beta_single fixed at 1.00.

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
EXP7_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${SRC_ROOT}:${EXP7_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

BETA_MULTIS=(0.85 0.90)
BETA_M=${BETA_MULTIS[$SLURM_ARRAY_TASK_ID]}
BETA_S=1.00

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayTask: ${SLURM_ARRAY_TASK_ID}  beta_multi=${BETA_M}  beta_single=${BETA_S}"

TEST_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface_lora/test"
SCALE4_DIR="${TEST_DIR}"
SCALE5_DIR="${TEST_DIR}"
SCALE6_DIR="${TEST_DIR}"

FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_index_arcface_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_labels_arcface_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp9b_knn_diff_beta_phase2a"

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
  --knn-k 20 \
  --knn-tau 0.07 \
  --betas "${BETA_M}" \
  --alphas "0.15" \
  --thresholds "0.100" \
  --scales "4,5,6" \
  --beta-singles "${BETA_S}" \
  --embedding-col "emb_arcface_f16" \
  --embedding-dim 768

echo "=== Done: beta_multi=${BETA_M}, beta_single=${BETA_S} ==="
echo "Reference: beta_multi=0.80, beta_single=1.00 -> F1=0.41976 (Tier 2 best so far)"
