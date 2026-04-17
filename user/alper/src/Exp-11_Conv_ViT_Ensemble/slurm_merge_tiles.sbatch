#!/bin/bash
# Step 1: Merge ViT + ConvNeXt tiles at w=0.78, save parquet

#SBATCH --job-name=merge-tiles
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_tiles_%A.out
#SBATCH --error=logs/merge_tiles_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

VIT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VIT_4x4="${VIT_DIR}/scale_4x4.parquet"
CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
OUT_PARQUET="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/merged_w0.78.parquet"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${EXP11_DIR}/logs"

echo "=== Merging tiles at w=0.78 ==="
echo "Date: $(date)"

python "${EXP11_DIR}/merge_tiles.py" \
    --vit-scales "${VIT_4x4}" "${VIT_DIR}/scale_5x5.parquet" "${VIT_DIR}/scale_6x6.parquet" \
    --cnxt-scales "${CNXT_DIR}/scale_4x4.parquet" "${CNXT_DIR}/scale_5x5.parquet" "${CNXT_DIR}/scale_6x6.parquet" \
    --blend-weight 0.78 \
    --out-parquet "${OUT_PARQUET}"

echo "=== Done ==="
