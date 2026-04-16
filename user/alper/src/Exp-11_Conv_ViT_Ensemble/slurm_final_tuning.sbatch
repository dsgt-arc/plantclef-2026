#!/bin/bash
# Exp-11: Final tuning — sweep threshold, ConvNeXt alpha, and fine weight

#SBATCH --job-name=final-tune
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_tuning_%A.out
#SBATCH --error=logs/final_tuning_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"
EXP8_DIR="${REPO_ROOT}/user/alper/src/Exp-8_Quantile_Aggregation_SS"

export PYTHONPATH="${EXP8_DIR}:${PYTHONPATH:-}"
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

VIT_KNN_PARQUET=$(find "${PLANTCLEF_ROOT}/aerten3/submissions/exp11_ensemble_knn/vit_knn_stage" -name "*fused*.parquet" | head -1)

BASE_OUT="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_final_tuning"

mkdir -p "${EXP11_DIR}/logs"

echo "=== Final Tuning ==="
echo "ViT+kNN parquet: ${VIT_KNN_PARQUET}"
echo "Date: $(date)"

CNXT_SCALES="${CNXT_DIR}/scale_4x4.parquet ${CNXT_DIR}/scale_5x5.parquet ${CNXT_DIR}/scale_6x6.parquet"

# 1. Threshold sweep: w=0.78, alpha=0.05, thr={0.085-0.110}
echo ""
echo "=== Sweep 1: Thresholds ==="
mkdir -p "${BASE_OUT}/thr_sweep"
python "${EXP11_DIR}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VIT_KNN_PARQUET}" \
    --convnext-scales ${CNXT_SCALES} \
    --species-map "${SPECIES_MAP}" --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${BASE_OUT}/thr_sweep" \
    --cnxt-alpha 0.05 \
    --weights "0.78" \
    --thresholds "0.085,0.090,0.095,0.105,0.110" \
    --topk 10

# 2. ConvNeXt alpha sweep: w=0.78, thr=0.100, alpha={0.10, 0.15}
echo ""
echo "=== Sweep 2: ConvNeXt alpha ==="
mkdir -p "${BASE_OUT}/alpha_sweep"
for ALPHA in 0.10 0.15; do
    echo "  alpha=${ALPHA}"
    python "${EXP11_DIR}/ensemble_with_knn.py" \
        --vit-knn-parquet "${VIT_KNN_PARQUET}" \
        --convnext-scales ${CNXT_SCALES} \
        --species-map "${SPECIES_MAP}" --class-mask-npy "${CLASS_MASK}" \
        --out-dir "${BASE_OUT}/alpha_sweep/a${ALPHA}" \
        --cnxt-alpha "${ALPHA}" \
        --weights "0.78" \
        --thresholds "0.100" \
        --topk 10
done

# 3. Fine weight sweep: thr=0.100, alpha=0.05, w={0.76, 0.77, 0.79}
echo ""
echo "=== Sweep 3: Fine weights ==="
mkdir -p "${BASE_OUT}/weight_sweep"
python "${EXP11_DIR}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VIT_KNN_PARQUET}" \
    --convnext-scales ${CNXT_SCALES} \
    --species-map "${SPECIES_MAP}" --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${BASE_OUT}/weight_sweep" \
    --cnxt-alpha 0.05 \
    --weights "0.76,0.77,0.79" \
    --thresholds "0.100" \
    --topk 10

echo ""
echo "=== All sweeps done ==="
echo "Results in:"
echo "  ${BASE_OUT}/thr_sweep/"
echo "  ${BASE_OUT}/alpha_sweep/"
echo "  ${BASE_OUT}/weight_sweep/"
