#!/bin/bash
#SBATCH --job-name=cnxt-p1
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase1_%x_%A.out
#SBATCH --error=logs/phase1_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# ---- Configuration ----
VARIANT="${VARIANT:-base}"
NGPUS=1  # 1 GPU sufficient for head-only training, faster scheduling

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext"

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs"

# ---- Info ----
echo "=== Phase 1: Head-only training ==="
echo "Variant: ${VARIANT}"
echo "GPUs: ${NGPUS}"
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import timm; print(f'timm {timm.__version__}')"

# ---- Train ----
if [ "${NGPUS}" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=${NGPUS} --master_port=$((29500 + RANDOM % 1000))"
else
    CMD="python"
fi

${CMD} "${EXP_DIR}/train_convnext.py" \
    -c "${EXP_DIR}/configs/phase1_${VARIANT}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    --output "${OUTPUT_DIR}" \
    --experiment "convnext_${VARIANT}_phase1"

echo "=== Phase 1 complete: ${VARIANT} ==="
echo "Best model: ${OUTPUT_DIR}/convnext_${VARIANT}_phase1/model_best.pth.tar"
