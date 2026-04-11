#!/bin/bash
# Phase 2 training with 4x H200 GPUs (test queue wait time)
# Saves to separate experiment dir to avoid collision with 2-GPU run

#SBATCH --job-name=cnxt-p2-4g
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/phase2_4gpu_%x_%A.out
#SBATCH --error=logs/phase2_4gpu_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# ---- Configuration ----
VARIANT="${VARIANT:-large_in22k}"
NGPUS=4
GRAD_ACCUM="${GRAD_ACCUM:-4}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext"

PHASE1_BEST="${OUTPUT_DIR}/convnext_${VARIANT}_phase1/model_best.pth.tar"

# ---- Validate Phase 1 checkpoint exists ----
if [ ! -f "${PHASE1_BEST}" ]; then
    echo "ERROR: Phase 1 best model not found at ${PHASE1_BEST}"
    exit 1
fi

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs"

# ---- Info ----
echo "=== Phase 2: Full fine-tuning (4x H200) ==="
echo "Variant: ${VARIANT}"
echo "GPUs: ${NGPUS}"
echo "Grad accum steps: ${GRAD_ACCUM}"
echo "Phase 1 checkpoint: ${PHASE1_BEST}"
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Train ----
LAST_CKPT="${OUTPUT_DIR}/convnext_${VARIANT}_phase2_4gpu/last.pth.tar"
if [ -f "${LAST_CKPT}" ]; then
    echo "Resuming from ${LAST_CKPT}"
    CHECKPOINT_ARG="--resume ${LAST_CKPT}"
else
    echo "Starting from Phase 1 checkpoint: ${PHASE1_BEST}"
    CHECKPOINT_ARG="--initial-checkpoint ${PHASE1_BEST}"
fi

torchrun --nproc_per_node=${NGPUS} --master_port=$((29500 + RANDOM % 1000)) \
    "${EXP_DIR}/train_convnext.py" \
    -c "${EXP_DIR}/configs/phase2_${VARIANT}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    ${CHECKPOINT_ARG} \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --output "${OUTPUT_DIR}" \
    --experiment "convnext_${VARIANT}_phase2_4gpu"

echo "=== Phase 2 complete: ${VARIANT} (4x H200) ==="
echo "Best model: ${OUTPUT_DIR}/convnext_${VARIANT}_phase2_4gpu/model_best.pth.tar"
