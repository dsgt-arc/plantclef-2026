#!/bin/bash
# Quick threshold sweep for ViT 4+5+6 above 0.100

#SBATCH --job-name=vit-thr
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vit_thr_%A.out
#SBATCH --error=logs/vit_thr_%A.err

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

VIT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VIT_4x4="${VIT_DIR}/scale_4x4.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_vit_thr_sweep"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

echo "=== ViT 4+5+6 threshold sweep ==="
python "${EXP11_DIR}/convnext_3x3_test.py" \
    --scale-files \
        "${VIT_4x4}" \
        "${VIT_DIR}/scale_5x5.parquet" \
        "${VIT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --tag "vit_456" \
    --thresholds "0.090,0.100,0.110,0.120,0.130,0.150"

echo "=== Done ==="
