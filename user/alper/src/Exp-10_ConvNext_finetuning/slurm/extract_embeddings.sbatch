#!/bin/bash
#SBATCH --job-name=cnxt-emb
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/embed_%x_%A_%a.out
#SBATCH --error=logs/embed_%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# ---- Configuration ----
VARIANT="${VARIANT:-base}"
SCALE="${SCALE:-9}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp10_convnext_${VARIANT}"

# Use the exported checkpoint (or model_best.pth.tar directly)
MODEL_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext/convnext_${VARIANT}_phase2/model_best.pth.tar"

# Map variant to timm model name
case "${VARIANT}" in
    tiny)  MODEL_NAME="convnext_tiny.dinov3_lvd1689m" ;;
    small) MODEL_NAME="convnext_small.dinov3_lvd1689m" ;;
    base)  MODEL_NAME="convnext_base.dinov3_lvd1689m" ;;
    large) MODEL_NAME="convnext_large.dinov3_lvd1689m" ;;
    *)     echo "Unknown variant: ${VARIANT}"; exit 1 ;;
esac

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs" "${OUTPUT_DIR}"

# ---- Info ----
echo "=== ConvNeXt Embedding Extraction ==="
echo "Variant: ${VARIANT} (${MODEL_NAME})"
echo "Scale: ${SCALE}x${SCALE}"
echo "Model: ${MODEL_PATH}"
echo "Images: ${IMAGE_DIR}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Extract ----
python "${EXP_DIR}/extract_convnext_embeddings.py" \
    --scale "${SCALE}" \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 64

echo "=== Done: scale ${SCALE}x${SCALE} ==="
