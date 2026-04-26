#!/bin/bash
#SBATCH --job-name=exp9b-diff-beta-2b
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

# Phase 2b: sweep beta_single below 1.00 with beta_multi=0.80 fixed.
# Tests whether the Tier 2 retriever is sharp enough to earn nonzero kNN
# contribution on single-visit images (which Exp-9 found to be hurt by kNN
# at any beta_single < 1.00 with the frozen DINOv2 retriever).
#
# This is the highest-potential remaining lever — if it works, beta_single
# drops below 1.00 and ensemble F1 lifts another +0.001 to +0.003 on top
# of the current 0.41976.
#
# Single job, sweeps 3 beta_single values internally (script's diff-beta mode).

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
echo "Job: ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"

TEST_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface_lora/test"
SCALE4_DIR="${TEST_DIR}"
SCALE5_DIR="${TEST_DIR}"
SCALE6_DIR="${TEST_DIR}"

FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_index_arcface_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_labels_arcface_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp9b_knn_diff_beta_phase2b"

# beta_multi=0.80 (current Tier 2 optimum), sweep beta_single in {0.85, 0.90, 0.95}.
# 1.00 already known: F1=0.41976 (no need to repeat).
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
  --betas "0.80" \
  --alphas "0.15" \
  --thresholds "0.100" \
  --scales "4,5,6" \
  --beta-singles "0.85,0.90,0.95" \
  --embedding-col "emb_arcface_f16" \
  --embedding-dim 768

echo "=== Phase 2b done ==="
echo "Reference: beta_multi=0.80, beta_single=1.00 -> F1=0.41976"
echo "Output:    ${OUT_DIR}"
