#!/bin/bash
# Approach A with ConvNeXt 4+5+6+7: ViT+kNN → blend with ConvNeXt 4567

#SBATCH --job-name=ens-knn-4567
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ens_knn_4567_%A.out
#SBATCH --error=logs/ens_knn_4567_%A.err
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_ens_knn_cnxt4567"

mkdir -p "${EXP11_DIR}/logs" "${OUT_DIR}"

echo "=== Approach A: ViT+kNN → blend with ConvNeXt 4+5+6+7 ==="
echo "ViT+kNN parquet: ${VIT_KNN_PARQUET}"
echo "Date: $(date)"

python "${EXP11_DIR}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VIT_KNN_PARQUET}" \
    --convnext-scales \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
        "${CNXT_DIR}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --cnxt-alpha 0.05 \
    --weights "0.75,0.78,0.80,0.82,0.85" \
    --thresholds "0.100" \
    --topk 10

echo "=== Done ==="
