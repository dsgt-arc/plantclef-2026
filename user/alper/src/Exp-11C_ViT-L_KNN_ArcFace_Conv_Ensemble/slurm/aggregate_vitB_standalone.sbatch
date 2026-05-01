#!/bin/bash
# Aggregate ViT-B existing multiscale tiles → submission CSVs for Kaggle
# Matched to aggregate_vitL_standalone.sbatch so ViT-L vs ViT-B is apples-to-apples
# (same schemes E,J,K,L, same alpha, same thresholds — only the model differs).
#
# Uses pre-extracted ViT-B test parquets from Exp-7 (no re-extraction needed).
#
# Usage:
#   sbatch slurm/aggregate_vitB_standalone.sbatch

#SBATCH --job-name=vitB-agg-standalone
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

# ViT-B parquet locations (already extracted in Exp-7)
VITB_MULTISCALE="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VITB_SCALE4="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitB_standalone"

# ViT-B model path (referenced by multiscale_aggregate.py for species map)
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

# ---- Validate ViT-B parquets exist ----
declare -A SCALE_DIRS
SCALE_DIRS[3]="${VITB_MULTISCALE}/scale_3x3.parquet"
SCALE_DIRS[4]="${VITB_SCALE4}"
SCALE_DIRS[5]="${VITB_MULTISCALE}/scale_5x5.parquet"
SCALE_DIRS[6]="${VITB_MULTISCALE}/scale_6x6.parquet"
SCALE_DIRS[7]="${VITB_MULTISCALE}/scale_7x7.parquet"

for s in 3 4 5 6 7; do
    if [ ! -e "${SCALE_DIRS[$s]}" ]; then
        echo "ERROR: missing ViT-B scale ${s}x${s} parquet at ${SCALE_DIRS[$s]}"
        exit 1
    fi
done

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-B standalone aggregation (matched to ViT-L sweep) ==="
echo "Output: ${OUT_DIR}"

python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale3-dir "${SCALE_DIRS[3]}" \
    --scale4-dir "${SCALE_DIRS[4]}" \
    --scale5-dir "${SCALE_DIRS[5]}" \
    --scale6-dir "${SCALE_DIRS[6]}" \
    --scale7-dir "${SCALE_DIRS[7]}" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --topk 10 \
    --image-topk 64 \
    --schemes "A,E,J,K,L" \
    --thresholds "0.075,0.100,0.125,0.150" \
    --csv-prefix "vitB_"

echo "=== Done. Submit CSVs in ${OUT_DIR} to Kaggle. ==="
echo "Existing Exp-7 ViT-B published numbers (different alpha tunings, NOT matched):"
echo "  Scheme A (4x4 only, 16 tiles): 0.35698 (Exp-3/4 baseline)"
echo "  Scheme E (4+5+6, 77 tiles)   : 0.39997 @ thr=0.100, alpha=0.10 (Exp-8 best)"
echo "                                 0.40100 @ thr=0.100, alpha=0.05 (Exp-8 final)"
echo "  Scheme J (3+4+5+6, 86 tiles) : 0.38227 @ thr=0.075, alpha=0.6"
echo "  Scheme K (4+5+6+7, 126 tiles): 0.38542 @ thr=0.080, alpha=0.6"
echo "  Scheme L (3+4+5+6+7, 135)    : not previously evaluated"
