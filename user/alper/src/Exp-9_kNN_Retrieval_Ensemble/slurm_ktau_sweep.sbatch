#!/bin/bash
#SBATCH --job-name=exp9-ktau-sweep
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9: Sweep kNN k and tau at fixed beta=0.80, alpha=0.15, thr=0.100.
# Each (k, tau) combo requires a full FAISS re-query (~30-60 min each).
# 5 k values × 5 tau values = 25 combos.

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9_kNN_Retrieval_Ensemble"
EXP8_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-8_Quantile_Aggregation"
export PYTHONPATH="${SRC_ROOT}:${EXP8_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

SCALE4_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
SCALE5_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
SCALE6_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"

FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_index_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_labels_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp9_knn_ktau_sweep"

echo "=== Exp-9: k/tau Sweep ==="
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
  --knn-k 20 \
  --knn-tau 0.07 \
  --betas "0.80" \
  --alphas "0.15" \
  --thresholds "0.100" \
  --scales "4,5,6" \
  --knn-ks "5,10,20,50,100" \
  --knn-taus "0.01,0.03,0.07,0.15,0.30"

echo ""
echo "=== k/tau sweep complete ==="
echo "Baseline: k=20, tau=0.07 → F1=0.41153"
