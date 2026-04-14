#!/bin/bash
# Extract ConvNeXt-L embeddings at 5x5 and 6x6 scales
# 4x4 already extracted — this completes the multi-scale set

#SBATCH --job-name=cnxt-ext56
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/extract_cnxt_56_%A.out
#SBATCH --error=logs/extract_cnxt_56_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP10_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

MODEL_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext/convnext_large_phase2_4gpu/model_best.pth.tar"
MODEL_NAME="convnext_large.dinov3_lvd1689m"

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

echo "=== ConvNeXt-L Multi-scale Extraction (5x5 + 6x6) ==="
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${MODEL_PATH}"
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Scale 5x5 (25 tiles/image) ----
echo ""
echo "=== Scale 5x5 ==="
python "${EXP10_DIR}/extract_convnext_embeddings.py" \
    --scale 5 \
    --tile-size 518 \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 16 \
    --output-filename "scale_5x5.parquet"

# ---- Scale 6x6 (36 tiles/image) ----
echo ""
echo "=== Scale 6x6 ==="
python "${EXP10_DIR}/extract_convnext_embeddings.py" \
    --scale 6 \
    --tile-size 518 \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 16 \
    --output-filename "scale_6x6.parquet"

echo ""
echo "=== All scales complete ==="
echo "Files in ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"/*.parquet
