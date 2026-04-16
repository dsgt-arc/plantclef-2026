#!/bin/bash
# Exp-11: Ensemble ViT+kNN with ConvNeXt
# Step 1: Run Exp-9 kNN pipeline on ViT → fused parquet
# Step 2: Blend ViT+kNN fused probs with ConvNeXt aggregated probs

#SBATCH --job-name=ens-knn
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ens_knn_%A.out
#SBATCH --error=logs/ens_knn_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP9_DIR="${REPO_ROOT}/user/alper/src/Exp-9_kNN_Retrieval_Ensemble"
EXP8_DIR="${REPO_ROOT}/user/alper/src/Exp-8_Quantile_Aggregation_SS"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

EXP8_DIR_ALT="${REPO_ROOT}/user/alper/src/Exp-8_Quantile_Aggregation"
export PYTHONPATH="${EXP9_DIR}:${EXP8_DIR}:${EXP8_DIR_ALT}:${PYTHONPATH:-}"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

# Paths
SCALE4_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
MULTISCALE_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"

FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_index_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_labels_geo.npy"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_ensemble_knn"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}/vit_knn_stage"

echo "=== Step 1: ViT + kNN pipeline (Exp-9 best config) ==="
echo "Date: $(date)"

python "${EXP9_DIR}/multiscale_aggregate_knn.py" \
    --scale4-dir "${SCALE4_DIR}" \
    --scale5-dir "${MULTISCALE_DIR}" \
    --scale6-dir "${MULTISCALE_DIR}" \
    --faiss-index "${FAISS_INDEX}" \
    --faiss-labels "${FAISS_LABELS}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}/vit_knn_stage" \
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
    --beta-singles "1.00"

# Find the ViT+kNN fused parquet
VIT_KNN_PARQUET=$(find "${OUT_DIR}/vit_knn_stage" -name "*fused*.parquet" | head -1)
echo ""
echo "ViT+kNN fused parquet: ${VIT_KNN_PARQUET}"

echo ""
echo "=== Step 2: Blend ViT+kNN with ConvNeXt ==="

python "${EXP11_DIR}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VIT_KNN_PARQUET}" \
    --convnext-scales \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --cnxt-alpha 0.05 \
    --weights "0.78,0.80,0.82,0.85,0.88,0.90,0.92,0.95" \
    --thresholds "0.100" \
    --topk 10

echo "=== Done ==="
