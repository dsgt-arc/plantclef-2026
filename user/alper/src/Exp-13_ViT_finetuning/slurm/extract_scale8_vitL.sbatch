#!/bin/bash
# Extract ViT-L test embeddings at scale 8x8 (64 tiles/image, 134,720 total tiles).
# Single-scale follow-up to extract_multiscale_3to7.sbatch.
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,EXP_SUFFIX=phase2v2 slurm/extract_scale8_vitL.sbatch

#SBATCH --job-name=vit-emb-s8
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/embed_%x_%A.out
#SBATCH --error=logs/embed_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

MODEL="${MODEL:-dinov2_large}"
EXP_SUFFIX="${EXP_SUFFIX:-phase2v2}"
OUTPUT_TAG="${OUTPUT_TAG:-${EXP_SUFFIX}}"
SCALE=8

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-13_ViT_finetuning"
IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit/vit_${MODEL}_${EXP_SUFFIX}/model_best.pth.tar"

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

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ViT-L checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs" "${OUTPUT_DIR}"

echo "=== ViT-L scale 8x8 extraction ==="
echo "Model:      ${MODEL} (${MODEL_NAME})"
echo "Suffix:     ${EXP_SUFFIX}"
echo "Tile size:  ${TILE_SIZE}"
echo "Checkpoint: ${MODEL_PATH}"
echo "Output:     ${OUTPUT_DIR}/scale_8x8.parquet"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "${EXP_DIR}/extract_vit_embeddings.py" \
    --scale "${SCALE}" \
    --tile-size "${TILE_SIZE}" \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 32

echo "=== Done: scale 8x8 ==="
