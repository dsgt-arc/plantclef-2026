#!/bin/bash
# Exp-10C — extract ConvNeXt-cooldown predictions at scales 5, 6, 7
# (scheme P, the production ensemble's ConvNeXt branch).
#
# Uses the phase2v3 cooldown checkpoint (last.pth.tar — model_best is
# contaminated because val == train_full during cooldown).
#
# Output: ${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large_phase2v3/
#   scale_5x5.parquet
#   scale_6x6.parquet
#   scale_7x7.parquet
#
# Usage:
#   sbatch --array=5-7 slurm/extract_phase2v3.sbatch
#
# Override checkpoint path:
#   sbatch --array=5-7 --export=CKPT=/path/to/last.pth.tar slurm/extract_phase2v3.sbatch

#SBATCH --job-name=cnxt-emb-p2v3
#SBATCH --account=paceship-clef2026_plantclef
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

VARIANT="${VARIANT:-large}"
SCALE="${SLURM_ARRAY_TASK_ID:-${SCALE:-5}}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_${VARIANT}_phase2v3"

# Cooldown checkpoint — use last.pth.tar (model_best is contaminated)
CKPT="${CKPT:-${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext/convnext_${VARIANT}_phase2v3/last.pth.tar}"

case "${VARIANT}" in
    tiny)  MODEL_NAME="convnext_tiny.dinov3_lvd1689m" ;;
    small) MODEL_NAME="convnext_small.dinov3_lvd1689m" ;;
    base)  MODEL_NAME="convnext_base.dinov3_lvd1689m" ;;
    large) MODEL_NAME="convnext_large.dinov3_lvd1689m" ;;
    *)     echo "Unknown variant: ${VARIANT}"; exit 1 ;;
esac

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: cooldown checkpoint not found at ${CKPT}"
    echo "Override with: sbatch --export=CKPT=/path/to/last.pth.tar ..."
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${REPO_ROOT}/user/alper/src/Exp-10C_ConvNext_Cooldown/logs" "${OUTPUT_DIR}"

echo "=== Exp-10C: ConvNeXt phase2v3 extraction ==="
echo "Variant:    ${VARIANT} (${MODEL_NAME})"
echo "Scale:      ${SCALE}x${SCALE}"
echo "Checkpoint: ${CKPT}"
echo "Images:     ${IMAGE_DIR}"
echo "Output:     ${OUTPUT_DIR}/scale_${SCALE}x${SCALE}.parquet"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "${EXP_DIR}/extract_convnext_embeddings.py" \
    --scale "${SCALE}" \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${CKPT}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 64

echo "=== Done: scale ${SCALE}x${SCALE} -> ${OUTPUT_DIR}/scale_${SCALE}x${SCALE}.parquet ==="
