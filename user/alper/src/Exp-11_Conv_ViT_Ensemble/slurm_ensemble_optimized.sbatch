#!/bin/bash
# Exp-11: Optimized ensemble — ConvNeXt 3+4+5+6+7 + ViT 4+5+6

#SBATCH --job-name=ens-opt
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ens_opt_%A.out
#SBATCH --error=logs/ens_opt_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
VIT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VIT_4x4="${VIT_DIR}/scale_4x4.parquet"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_ensemble_optimized"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

echo "=== Optimized Ensemble: CNxt 3+4+5+6+7 + ViT 4+5+6 ==="
echo "Date: $(date)"

python "${EXP11_DIR}/ensemble_optimized.py" \
    --convnext-scales \
        "${CNXT_DIR}/scale_3x3.parquet" \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
        "${CNXT_DIR}/scale_7x7.parquet" \
    --vit-scales \
        "${VIT_4x4}" \
        "${VIT_DIR}/scale_5x5.parquet" \
        "${VIT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.05 \
    --weights "0.75,0.78,0.80,0.82,0.85" \
    --thresholds "0.090,0.100,0.110,0.120" \
    --topk 10

echo "=== Done ==="
