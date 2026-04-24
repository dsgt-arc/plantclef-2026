#!/bin/bash
#SBATCH --job-name=exp9b-knn-sweep
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B: Standard beta/alpha/threshold sweep using ArcFace retriever.
# Adapted from Exp-9/slurm_knn_sweep.sbatch — paths point to 512-d ArcFace
# index, test parquets are the projected versions carrying emb_arcface_f16.
#
# CPU-only job. 512-d FAISS index loaded into RAM (~1.7 GB vs 2.5 GB for 768-d).

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
EXP8_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-8_Quantile_Aggregation"
export PYTHONPATH="${SRC_ROOT}:${EXP8_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

# Projected test parquets — produced by project_embeddings.py --mode test.
# Flat layout: scale_{4x4,5x5,6x6}.parquet all under this directory with
# the original columns preserved plus the new emb_arcface_f16 column.
TEST_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/test"
SCALE4_DIR="${TEST_DIR}"
SCALE5_DIR="${TEST_DIR}"
SCALE6_DIR="${TEST_DIR}"

# 512-d ArcFace FAISS index
FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_index_arcface_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_labels_arcface_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp9b_knn_retrieval"

KNN_K=20
KNN_TAU=0.07

echo "=== Exp-9B: ArcFace kNN Retrieval Ensemble Sweep ==="
echo "FAISS index: ${FAISS_INDEX}"
echo "kNN k=${KNN_K}, tau=${KNN_TAU}"
echo "Output: ${OUT_DIR}"

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
  --knn-k "${KNN_K}" \
  --knn-tau "${KNN_TAU}" \
  --betas "0.80" \
  --alphas "0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60" \
  --thresholds "0.040,0.060,0.070,0.080,0.090,0.100,0.110" \
  --scales "4,5,6" \
  --embedding-col "emb_arcface_f16" \
  --embedding-dim 512

echo ""
echo "=== Exp-9B sweep complete ==="
echo "All CSVs in: ${OUT_DIR}"
echo "Exp-9 baseline (768-d DINOv2 retriever): beta=0.80, alpha=0.15, thr=0.100 -> F1=0.41153"
