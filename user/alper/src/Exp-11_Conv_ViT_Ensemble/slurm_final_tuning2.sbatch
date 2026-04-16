#!/bin/bash
# Exp-11: Final tuning round 2 — combine best weight + alpha sweep

#SBATCH --job-name=final-tune2
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/final_tuning2_%A.out
#SBATCH --error=logs/final_tuning2_%A.err
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

BASE_OUT="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_final_tuning2"
CNXT_SCALES="${CNXT_DIR}/scale_4x4.parquet ${CNXT_DIR}/scale_5x5.parquet ${CNXT_DIR}/scale_6x6.parquet"

mkdir -p "${EXP11_DIR}/logs"

echo "=== Final Tuning Round 2: w=0.77, alpha sweep ==="
echo "Date: $(date)"

for ALPHA in 0.15 0.20 0.25; do
    echo ""
    echo "=== w=0.77, alpha=${ALPHA} ==="
    mkdir -p "${BASE_OUT}/w0.77_a${ALPHA}"
    python "${EXP11_DIR}/ensemble_with_knn.py" \
        --vit-knn-parquet "${VIT_KNN_PARQUET}" \
        --convnext-scales ${CNXT_SCALES} \
        --species-map "${SPECIES_MAP}" --class-mask-npy "${CLASS_MASK}" \
        --out-dir "${BASE_OUT}/w0.77_a${ALPHA}" \
        --cnxt-alpha "${ALPHA}" \
        --weights "0.77" \
        --thresholds "0.100" \
        --topk 10
done

echo ""
echo "=== Done ==="
