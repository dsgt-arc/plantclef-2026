#!/bin/bash
#SBATCH --job-name=exp4-extract-views
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-4_Enhanced_Tile_Aggregation"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID:-}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi || true

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
OUTPUT_PATH="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_additional_views/additional_views.parquet"

echo "=== Step 6: Extract additional augmentation views (rot180, rot270, vflip) ==="
echo "Image dir: ${IMAGE_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"

python "${SRC_ROOT}/extract_additional_views.py" \
  --image-dir "${IMAGE_DIR}" \
  --output-path "${OUTPUT_PATH}" \
  --model-path "${MODEL_PATH}" \
  --batch-size 16

echo ""
echo "=== Extraction complete ==="
echo "Output: ${OUTPUT_PATH}"
echo "Next: run slurm_sixview_sweep.sbatch for 6-view aggregation"
