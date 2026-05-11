#!/bin/bash
# Extract ViT-L phase2v3 test embeddings at scales {1x1, 2x2, 3x3, 8x8}.
# Companion to extract_multiscale_3to7.sbatch (which produced 3-7) and
# extract_scale8_vitL.sbatch — extends the coverage to extreme tile counts
# for later experiments (single-tile baseline, fine-grained 8x8).
#
# Note: scale 3x3 is also re-extracted here; existing scale_3x3.parquet
# at the same output path will be overwritten (output is deterministic
# given the same model + images + JPEG quality).
#
# Tile counts (test set has 2105 images):
#   scale 1: 1  tile  per image →   2,105 tiles  (single-tile baseline)
#   scale 2: 4  tiles per image →   8,420 tiles
#   scale 3: 9  tiles per image →  18,945 tiles
#   scale 8: 64 tiles per image → 134,720 tiles  (dominates wall-time)
#
# Usage:
#   sbatch slurm/extract_scales_1_2_3_8_vitL_phase2v3.sbatch
#
#   # Smaller batch for scale 8 on RTX-6000 (24GB) if OOM occurs:
#   sbatch --export=BATCH_SIZE=16 slurm/extract_scales_1_2_3_8_vitL_phase2v3.sbatch
#
# Output: ${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_dinov2_large_phase2v3/scale_NxN.parquet

#SBATCH --job-name=vit-emb-extra
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=1,2,3,8
#SBATCH --output=logs/embed_%x_%A_%a.out
#SBATCH --error=logs/embed_%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# Fixed model/checkpoint (production phase2v3 cooldown ViT-L)
MODEL="dinov2_large"
EXP_SUFFIX="phase2v3"
OUTPUT_TAG="phase2v3"
SCALE="${SLURM_ARRAY_TASK_ID}"
BATCH_SIZE="${BATCH_SIZE:-32}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-13_ViT_finetuning"
IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"
MODEL_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit/vit_${MODEL}_${EXP_SUFFIX}/model_best.pth.tar"

# DINOv2 ViT-L: patch14, native tile_size=518
MODEL_NAME="vit_large_patch14_reg4_dinov2.lvd142m"
TILE_SIZE=518

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: ViT-L checkpoint not found at ${MODEL_PATH}"
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs" "${OUTPUT_DIR}"

echo "=== ViT-L phase2v3 extraction (SCALE=${SCALE}) ==="
echo "Model:      ${MODEL} (${MODEL_NAME})"
echo "Suffix:     ${EXP_SUFFIX}"
echo "Scale:      ${SCALE}x${SCALE}"
echo "Tile size:  ${TILE_SIZE}"
echo "Batch:      ${BATCH_SIZE}"
echo "Checkpoint: ${MODEL_PATH}"
echo "Images:     ${IMAGE_DIR}"
echo "Output:     ${OUTPUT_DIR}/scale_${SCALE}x${SCALE}.parquet"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "${EXP_DIR}/extract_vit_embeddings.py" \
    --scale "${SCALE}" \
    --tile-size "${TILE_SIZE}" \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size "${BATCH_SIZE}"

echo "=== Done: scale ${SCALE}x${SCALE} ==="
