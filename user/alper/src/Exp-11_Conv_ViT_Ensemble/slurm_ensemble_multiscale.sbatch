#!/bin/bash
# Exp-11: Full-pipeline multi-scale ensemble (ConvNeXt + ViT)
# Both models: 4x4+5x5+6x6 → max_prob → temporal fusion → blend → submission

#SBATCH --job-name=ms-ensemble
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ms_ensemble_%A.out
#SBATCH --error=logs/ms_ensemble_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

# ConvNeXt parquets (from extraction jobs)
CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"

# ViT parquets
VIT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VIT_4x4="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale/scale_4x4.parquet"

# Mapping and masking
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

# Output
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_multiscale_ensemble"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

echo "=== Exp-11: Multi-scale Ensemble ==="
echo "Date: $(date)"

python "${EXP11_DIR}/ensemble_multiscale.py" \
    --convnext-4x4 "${CNXT_DIR}/scale_4x4.parquet" \
    --convnext-5x5 "${CNXT_DIR}/scale_5x5.parquet" \
    --convnext-6x6 "${CNXT_DIR}/scale_6x6.parquet" \
    --vit-4x4 "${VIT_4x4}" \
    --vit-5x5 "${VIT_DIR}/scale_5x5.parquet" \
    --vit-6x6 "${VIT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.05 \
    --weights "0.82,0.85,0.87,0.9,0.93,0.95" \
    --thresholds "0.075,0.100,0.125" \
    --topk 10

echo "=== Done ==="
