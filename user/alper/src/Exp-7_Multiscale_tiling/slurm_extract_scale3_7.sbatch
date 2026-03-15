#!/bin/bash
#SBATCH --job-name=exp7-extract-scale3-7
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Extract DINOv2 embeddings at scales 3x3 and 7x7 with JPEG85 compression.
# Array job: task 0=scale3 (9 tiles, ~15min), task 1=scale7 (49 tiles, ~4hrs)
#
# 3x3: 1000x1000px per tile, downscaled to 518 — between 2x2 and 4x4
# 7x7: ~430x430px per tile, upscaled to 518 — finer than 6x6

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayJobID: ${SLURM_ARRAY_JOB_ID:-}  TaskID: ${SLURM_ARRAY_TASK_ID:-}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi || true

# Map array task ID to scale
SCALES=(3 7)
SCALE=${SCALES[${SLURM_ARRAY_TASK_ID}]}

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"

echo "=== Extracting embeddings at scale ${SCALE}x${SCALE} ==="
echo "JPEG: quality=85, subsampling=1 (4:2:2)"
echo "Output: ${OUTPUT_DIR}"

python "${SRC_ROOT}/extract_multiscale_embeddings.py" \
  --scale "${SCALE}" \
  --image-dir "${IMAGE_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --jpeg-quality 85 \
  --jpeg-subsampling 1 \
  --batch-size 16

echo "Done. Scale ${SCALE}x${SCALE} extraction complete."
