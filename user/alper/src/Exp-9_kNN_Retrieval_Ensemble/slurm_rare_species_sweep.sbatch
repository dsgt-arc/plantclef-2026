#!/bin/bash
#SBATCH --job-name=exp9-rare-spp
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9: Rare species boosting strategies.
#
# Array index maps to:
#   0: baseline (no boosting, should reproduce F1=0.41153)
#   1: inverse frequency weighting
#   2: confidence gating at 0.3
#   3: confidence gating at 0.5
#   4: inv-freq + confidence gate 0.3
#   5: beta_single=0.90 (less kNN for single-visit)

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
echo "Job: ${SLURM_JOB_NAME}  ArrayTask: ${SLURM_ARRAY_TASK_ID}"

SCALE4_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
SCALE5_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
SCALE6_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"

FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_index_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_labels_geo.npy"
FREQ_WEIGHTS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/class_counts_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp9_knn_rare_species"

# Common args
COMMON="--scale4-dir ${SCALE4_DIR} --scale5-dir ${SCALE5_DIR} --scale6-dir ${SCALE6_DIR} \
  --faiss-index ${FAISS_INDEX} --faiss-labels ${FAISS_LABELS} \
  --species-map ${SPECIES_MAP} --class-mask-npy ${GEO_MASK} \
  --out-dir ${OUT_DIR} \
  --temperature 1.5 --topk 10 --image-topk 64 --tile-filter-k 0 \
  --knn-k 20 --knn-tau 0.07 \
  --betas 0.80 --alphas 0.15 --thresholds 0.100 --scales 4,5,6"

case ${SLURM_ARRAY_TASK_ID} in
  0)
    echo "=== Baseline (no boosting) ==="
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" ${COMMON}
    ;;
  1)
    echo "=== Inverse frequency weighting ==="
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" ${COMMON} \
      --freq-weights "${FREQ_WEIGHTS}" --use-inv-freq
    ;;
  2)
    echo "=== Confidence gating at 0.3 ==="
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" ${COMMON} \
      --confidence-gate 0.3
    ;;
  3)
    echo "=== Confidence gating at 0.5 ==="
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" ${COMMON} \
      --confidence-gate 0.5
    ;;
  4)
    echo "=== Inv-freq + confidence gate 0.3 ==="
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" ${COMMON} \
      --freq-weights "${FREQ_WEIGHTS}" --use-inv-freq --confidence-gate 0.3
    ;;
  5)
    echo "=== beta_single=0.90 (less kNN for single-visit) ==="
    python "${SRC_ROOT}/multiscale_aggregate_knn.py" ${COMMON} \
      --beta-singles "0.90"
    ;;
esac

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
