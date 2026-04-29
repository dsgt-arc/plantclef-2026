#!/bin/bash
# ViT-L follow-up: higher-temperature sweep
# Tests whether flattening ViT-L's peaked softmax via T=2.0 / T=2.5 lets more
# secondary species cross the threshold and improves multi-label F1.
#
# Schemes: E (456, 77 tiles) and L (34567, 135 tiles).
# Temperatures: 2.0 and 2.5 (T=1.5 was the ViT-B optimum).
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/aggregate_vitL_highT.sbatch

#SBATCH --job-name=vitL-highT
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_highT_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

# ---- Validate scale parquets ----
for s in 3 4 5 6 7; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L high-temperature sweep ==="
echo "Output: ${OUT_DIR}"

# multiscale_aggregate.py runs one temperature at a time → invoke twice
for T in 2.0 2.5; do
    echo ""
    echo "--- T=${T} ---"
    python "${SRC_ROOT}/multiscale_aggregate.py" \
        --scale3-dir "${PARQUET_ROOT}/scale_3x3.parquet" \
        --scale4-dir "${PARQUET_ROOT}/scale_4x4.parquet" \
        --scale5-dir "${PARQUET_ROOT}/scale_5x5.parquet" \
        --scale6-dir "${PARQUET_ROOT}/scale_6x6.parquet" \
        --scale7-dir "${PARQUET_ROOT}/scale_7x7.parquet" \
        --model-path "${MODEL_PATH}" \
        --species-map "${SPECIES_MAP}" \
        --class-mask-npy "${GEO_MASK}" \
        --out-dir "${OUT_DIR}" \
        --temperature "${T}" \
        --alpha 0.15 \
        --topk 10 \
        --image-topk 64 \
        --schemes "E,L" \
        --thresholds "0.075,0.100,0.125" \
        --csv-prefix "vitL_"
done

echo "=== Done. 12 CSVs in ${OUT_DIR} (2 schemes × 2 temps × 3 thresholds). ==="
echo "Compare to first-sweep ViT-L scheme E T=1.5 thr=0.100 (=0.38364)."
