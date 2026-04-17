#!/bin/bash
# Test multiple tiling schemes for ConvNeXt standalone

#SBATCH --job-name=cnxt-scales
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cnxt_scale_test_%A.out
#SBATCH --error=logs/cnxt_scale_test_%A.err

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_cnxt_scale_test"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

# Scheme: 3+4+5+6+7 (add 7x7 to best scheme)
echo "=== Scheme: 3+4+5+6+7 ==="
python "${EXP11_DIR}/convnext_3x3_test.py" \
    --scale-files \
        "${CNXT_DIR}/scale_3x3.parquet" \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
        "${CNXT_DIR}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --tag "34567" \
    --thresholds "0.100"

# Scheme: 2+3+4+5+6 (add 2x2)
echo "=== Scheme: 2+3+4+5+6 ==="
python "${EXP11_DIR}/convnext_3x3_test.py" \
    --scale-files \
        "${CNXT_DIR}/scale_2x2.parquet" \
        "${CNXT_DIR}/scale_3x3.parquet" \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --tag "23456" \
    --thresholds "0.100"

# Scheme: 1+2+3+4+5+6 (add 1x1)
echo "=== Scheme: 1+2+3+4+5+6 ==="
python "${EXP11_DIR}/convnext_3x3_test.py" \
    --scale-files \
        "${CNXT_DIR}/scale_1x1.parquet" \
        "${CNXT_DIR}/scale_2x2.parquet" \
        "${CNXT_DIR}/scale_3x3.parquet" \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --tag "123456" \
    --thresholds "0.100"

# Scheme: 2+3+4+5+6+7 (all except 1x1)
echo "=== Scheme: 2+3+4+5+6+7 ==="
python "${EXP11_DIR}/convnext_3x3_test.py" \
    --scale-files \
        "${CNXT_DIR}/scale_2x2.parquet" \
        "${CNXT_DIR}/scale_3x3.parquet" \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
        "${CNXT_DIR}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --tag "234567" \
    --thresholds "0.100"

# Scheme: 4+5+6+7 (add 7x7 to original best)
echo "=== Scheme: 4+5+6+7 ==="
python "${EXP11_DIR}/convnext_3x3_test.py" \
    --scale-files \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
        "${CNXT_DIR}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --tag "4567" \
    --thresholds "0.100"

echo "=== All schemes done ==="
