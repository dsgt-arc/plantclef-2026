#!/bin/bash
# Exp-11 Approach C: Full kNN sweep on pre-merged ensemble tiles
# Each job: one (k, tau) combo, sweeps all betas
# Usage: sbatch --export=KNN_K=20,KNN_TAU=0.07 slurm_knn_full_sweep.sbatch

#SBATCH --job-name=knn-sweep
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/knn_sweep_%A.out
#SBATCH --error=logs/knn_sweep_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

KNN_K="${KNN_K:-20}"
KNN_TAU="${KNN_TAU:-0.07}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"
EXP9_DIR="${REPO_ROOT}/user/alper/src/Exp-9_kNN_Retrieval_Ensemble"
EXP8_DIR="${REPO_ROOT}/user/alper/src/Exp-8_Quantile_Aggregation_SS"
EXP8_DIR_ALT="${REPO_ROOT}/user/alper/src/Exp-8_Quantile_Aggregation"

export PYTHONPATH="${EXP9_DIR}:${EXP8_DIR}:${EXP8_DIR_ALT}:${PYTHONPATH:-}"
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

MERGED="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/merged_w0.78.parquet"
FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_index_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_labels_geo.npy"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_knn_full_sweep"

mkdir -p "${EXP11_DIR}/logs" "${OUT_DIR}"

echo "=== kNN sweep: k=${KNN_K}, tau=${KNN_TAU} ==="
echo "Date: $(date)"

python "${EXP11_DIR}/run_knn_on_merged.py" \
    --merged-parquet "${MERGED}" \
    --faiss-index "${FAISS_INDEX}" \
    --faiss-labels "${FAISS_LABELS}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --knn-k "${KNN_K}" \
    --knn-tau "${KNN_TAU}" \
    --knn-beta-multis "0.80,0.85,0.90,0.95" \
    --knn-beta-single 1.00 \
    --threshold 0.100 \
    --topk 10 \
    --tag "w0.78"

echo "=== Done: k=${KNN_K}, tau=${KNN_TAU} ==="
