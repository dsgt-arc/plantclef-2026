#!/bin/bash
# Extract ViT-L test embeddings at 5 multiscale scales (3x3, 4x4, 5x5, 6x6, 7x7)
# as a SLURM array job. Each task handles one scale.
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,EXP_SUFFIX=phase2v2 slurm/extract_multiscale_3to7.sbatch
#
# Env vars:
#   MODEL       : dinov2_large | dinov3_large
#   EXP_SUFFIX  : phase2 | phase2v2 (which checkpoint dir to use)
#   OUTPUT_TAG  : optional, defaults to ${EXP_SUFFIX}
#
# Output: /storage/.../aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}/scale_NxN.parquet

#SBATCH --job-name=vit-emb-ms
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=3-7
#SBATCH --output=logs/embed_%x_%A_%a.out
#SBATCH --error=logs/embed_%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# ---- Configuration ----
MODEL="${MODEL:-dinov2_large}"
EXP_SUFFIX="${EXP_SUFFIX:-phase2v2}"
OUTPUT_TAG="${OUTPUT_TAG:-${EXP_SUFFIX}}"
SCALE="${SLURM_ARRAY_TASK_ID}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-13_ViT_finetuning"
IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit/vit_${MODEL}_${EXP_SUFFIX}/model_best.pth.tar"

# Map short model id to timm model name and tile size
case "${MODEL}" in
    dinov2_large)
        MODEL_NAME="vit_large_patch14_reg4_dinov2.lvd142m"
        TILE_SIZE=518
        ;;
    dinov3_large)
        MODEL_NAME="vit_large_patch16_dinov3.lvd1689m"
        TILE_SIZE=512
        ;;
    *)
        echo "Unknown MODEL: ${MODEL}"
        exit 1
        ;;
esac

# ---- Validate checkpoint ----
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ViT-L checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs" "${OUTPUT_DIR}"

# ---- Info ----
echo "=== ViT-L Multi-scale Extraction (array task SCALE=${SCALE}) ==="
echo "Model:      ${MODEL} (${MODEL_NAME})"
echo "Suffix:     ${EXP_SUFFIX}"
echo "Scale:      ${SCALE}x${SCALE}"
echo "Tile size:  ${TILE_SIZE}"
echo "Checkpoint: ${MODEL_PATH}"
echo "Images:     ${IMAGE_DIR}"
echo "Output:     ${OUTPUT_DIR}/scale_${SCALE}x${SCALE}.parquet"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Extract ----
python "${EXP_DIR}/extract_vit_embeddings.py" \
    --scale "${SCALE}" \
    --tile-size "${TILE_SIZE}" \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 32

echo "=== Done: scale ${SCALE}x${SCALE} ==="
