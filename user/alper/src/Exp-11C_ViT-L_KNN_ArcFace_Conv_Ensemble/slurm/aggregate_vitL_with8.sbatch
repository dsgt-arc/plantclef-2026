#!/bin/bash
# ViT-L aggregation including new 8x8 scale.
# Requires: scale_8x8.parquet (extracted by Exp-13/slurm/extract_scale8_vitL.sbatch)
#
# Tests three new schemes that add 8x8 to the existing best:
#   M = 4+5+6+7+8       (190 tiles) — extends K with 8
#   N = 3+4+5+6+7+8     (199 tiles) — extends L with 8 (full 6-scale stack)
#   O = 3+4+5+6+8       (150 tiles) — extends J with 8 (skip 7)
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/aggregate_vitL_with8.sbatch

#SBATCH --job-name=vitL-with8
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

MODEL="${MODEL:-dinov2_large}"
OUTPUT_TAG="${OUTPUT_TAG:-phase2v2}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_with8_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

# Scale 8 must exist
for s in 3 4 5 6 7 8; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        if [ "$s" = "8" ]; then
            echo "Run extract_scale8_vitL.sbatch first."
        fi
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L aggregation including 8x8 (schemes M,N,O) ==="
echo "Output: ${OUT_DIR}"
echo ""
echo "Threshold strategy:"
echo "  M (no 3x3, has 8x8) -> thr in {0.075,0.080,0.085,0.090,0.100}"
echo "  N,O (with 3x3+8x8)  -> thr in {0.080,0.090,0.100,0.115,0.130} (3x3 shifts optimum higher)"

# Scheme M (no 3x3): tighter range around 0.080 (the no-3x3 ViT-L optimum)
python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale3-dir "${PARQUET_ROOT}/scale_3x3.parquet" \
    --scale4-dir "${PARQUET_ROOT}/scale_4x4.parquet" \
    --scale5-dir "${PARQUET_ROOT}/scale_5x5.parquet" \
    --scale6-dir "${PARQUET_ROOT}/scale_6x6.parquet" \
    --scale7-dir "${PARQUET_ROOT}/scale_7x7.parquet" \
    --scale8-dir "${PARQUET_ROOT}/scale_8x8.parquet" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --topk 10 \
    --image-topk 64 \
    --schemes "M" \
    --thresholds "0.075,0.080,0.085,0.090,0.100" \
    --csv-prefix "vitL_"

# Schemes N, O (with 3x3): expanded range up through 0.130
python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale3-dir "${PARQUET_ROOT}/scale_3x3.parquet" \
    --scale4-dir "${PARQUET_ROOT}/scale_4x4.parquet" \
    --scale5-dir "${PARQUET_ROOT}/scale_5x5.parquet" \
    --scale6-dir "${PARQUET_ROOT}/scale_6x6.parquet" \
    --scale7-dir "${PARQUET_ROOT}/scale_7x7.parquet" \
    --scale8-dir "${PARQUET_ROOT}/scale_8x8.parquet" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --topk 10 \
    --image-topk 64 \
    --schemes "N,O" \
    --thresholds "0.080,0.090,0.100,0.115,0.130" \
    --csv-prefix "vitL_"

echo "=== Done. 15 CSVs in ${OUT_DIR} (M:5 + N:5 + O:5 = 15). ==="
echo "Reference: vitL scheme L thr=0.080 alpha=0.15 = 0.40945 (best so far without 8x8)."
echo "Note: still using alpha=0.15. If alpha sweep finds a better alpha, re-run this with that value."
