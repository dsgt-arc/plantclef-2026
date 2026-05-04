#!/bin/bash
# Phase2v3 standalone: test new scheme P (5+6+7, 110 tiles).
#
# Drops scale 4 from K (4+5+6+7, 126 tiles) to test whether the
# coarse-area low-resolution tiles are diluting the signal. Pattern from
# Phase 0:
#   E (4+5+6, 77)  → 0.39700
#   K (4+5+6+7, 126) → 0.40166 → 0.41186 tuned
#   L (3+4+5+6+7, 135) → 0.40114 (tied with K, scale 3 doesn't help)
#
# If P beats K, the scale-4 contribution at K is net-negative.
#
# Usage:
#   sbatch slurm/aggregate_vitL_P_phase2v3.sbatch

#SBATCH --job-name=vitL-P-p2v3
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

MODEL="dinov2_large"
OUTPUT_TAG="phase2v3"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_P_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

# Scheme P uses scales 5, 6, 7
for s in 5 6 7; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== Phase2v3 ViT-L scheme P (5+6+7, 110 tiles) ==="
echo "Reference K best: alpha=0.30, thr=0.085 = 0.41186"
echo "Output: ${OUT_DIR}"

# 2 alphas × 3 thresholds = 6 CSVs at K's locked T=1.5
for ALPHA in 0.20 0.30; do
    echo ""
    echo "--- alpha=${ALPHA} ---"
    python "${SRC_ROOT}/multiscale_aggregate.py" \
        --scale5-dir "${PARQUET_ROOT}/scale_5x5.parquet" \
        --scale6-dir "${PARQUET_ROOT}/scale_6x6.parquet" \
        --scale7-dir "${PARQUET_ROOT}/scale_7x7.parquet" \
        --scale4-dir "${PARQUET_ROOT}/scale_4x4.parquet" \
        --model-path "${MODEL_PATH}" \
        --species-map "${SPECIES_MAP}" \
        --class-mask-npy "${GEO_MASK}" \
        --out-dir "${OUT_DIR}" \
        --temperature 1.5 \
        --alpha "${ALPHA}" \
        --topk 10 \
        --image-topk 64 \
        --schemes "P" \
        --thresholds "0.080,0.085,0.095" \
        --csv-prefix "vitL_p2v3_a${ALPHA}_"
done

echo ""
echo "=== Done. 6 CSVs in ${OUT_DIR} ==="
echo "Submission strategy:"
echo "  Round 1 (2 LB): alpha=0.30 thr=0.085, alpha=0.20 thr=0.085 (matches K's best config)"
echo "  Round 2 (2 LB if P competitive): expand thresholds at winning alpha"
echo ""
echo "Decision:"
echo "  P > K + 0.005 → switch standalone winner to P (saves 13% inference compute)"
echo "  P within ±0.005 of K → keep K (it has the inference data already extracted for ensemble)"
echo "  P < K - 0.005 → close the question, K wins"
