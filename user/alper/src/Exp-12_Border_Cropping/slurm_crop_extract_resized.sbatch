#!/bin/bash
#SBATCH --job-name=exp12-crop-resized
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-12 Control: Crop borders + RESIZE BACK to original dimensions.
# Isolates the content-removal effect from the dimension-change effect.
#
# Array job: task 0=3%, task 1=5%, task 2=7%
# Each task: crop at N%, resize back to original size, extract 4x4 embeddings.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

EXP12_SRC="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-12_Border_Cropping"
EXP7_SRC="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${EXP7_SRC}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayJobID: ${SLURM_ARRAY_JOB_ID:-}  TaskID: ${SLURM_ARRAY_TASK_ID:-}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi || true

# Map array task ID to crop percentage
CROP_PCTS=(3 5 7)
CROP_PCT=${CROP_PCTS[${SLURM_ARRAY_TASK_ID}]}

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
CROPPED_ROOT="${PLANTCLEF_ROOT}/aerten3/data/exp12_cropped_images"
CROPPED_DIR="${CROPPED_ROOT}/crop_${CROP_PCT}pct_resized"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp12_border_crop"

echo ""
echo "=== Step 1: Crop at ${CROP_PCT}% + resize back to original dimensions ==="
echo "Input:  ${IMAGE_DIR}"
echo "Output: ${CROPPED_DIR}"

python "${EXP12_SRC}/crop_test_images.py" \
  --image-dir "${IMAGE_DIR}" \
  --output-root "${CROPPED_ROOT}" \
  --crop-pcts "${CROP_PCT}" \
  --jpeg-quality 95 \
  --resize-back

echo ""
echo "=== Step 2: Extract 4x4 DINOv2 embeddings (JPEG85) ==="
echo "Input:  ${CROPPED_DIR}"
echo "Output: ${OUTPUT_DIR}"

python "${EXP7_SRC}/extract_multiscale_embeddings.py" \
  --scale 4 \
  --image-dir "${CROPPED_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --jpeg-quality 85 \
  --jpeg-subsampling 1 \
  --batch-size 16 \
  --output-filename "scale_4x4_crop${CROP_PCT}pct_resized.parquet"

echo ""
echo "Done. Crop ${CROP_PCT}% (resized back) extraction complete."
echo "Parquet: ${OUTPUT_DIR}/scale_4x4_crop${CROP_PCT}pct_resized.parquet"
