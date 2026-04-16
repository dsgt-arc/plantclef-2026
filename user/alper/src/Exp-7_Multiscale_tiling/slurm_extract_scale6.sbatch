#!/bin/bash
#SBATCH --job-name=exp7-extract-scale6
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Extract DINOv2 embeddings at scale 6x6 with JPEG85 compression.
# 36 tiles/image x 2105 images = 75,780 inferences. Estimated ~3 hrs.
#
# Note: 6x6 means images are resized to 3108x3108 before cutting.
# For ~3000x3000 source images this is slight upscaling (~3.6%).

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi || true

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"

echo "=== Extracting embeddings at scale 6x6 ==="
echo "JPEG: quality=85, subsampling=1 (4:2:2)"
echo "Output: ${OUTPUT_DIR}"

python "${SRC_ROOT}/extract_multiscale_embeddings.py" \
  --scale 6 \
  --image-dir "${IMAGE_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --jpeg-quality 85 \
  --jpeg-subsampling 1 \
  --batch-size 16

echo "Done. Scale 6x6 extraction complete."
