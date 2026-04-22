#!/bin/bash
# Phase 2 full fine-tuning for ViT-Large
# Usage:
#   sbatch --export=MODEL=dinov2_large slurm/phase2_train.sbatch
#   sbatch --export=MODEL=dinov3_large slurm/phase2_train.sbatch

#SBATCH --job-name=vit-p2
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=logs/phase2_%x_%A.out
#SBATCH --error=logs/phase2_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# ---- Configuration ----
# MODEL: dinov2_large | dinov3_large
MODEL="${MODEL:-dinov2_large}"
NGPUS="${NGPUS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
# USE_EMA_INIT=1 loads state_dict_ema from Phase 1 best checkpoint when starting
# (recommended when P1 was stopped before full convergence; default on).
USE_EMA_INIT="${USE_EMA_INIT:-1}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-13_ViT_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit"

PHASE1_BEST="${PHASE1_BEST:-${OUTPUT_DIR}/vit_${MODEL}_phase1/model_best.pth.tar}"
EXP_SUFFIX="${EXP_SUFFIX:-phase2}"

# ---- Validate Phase 1 checkpoint exists ----
if [ ! -f "${PHASE1_BEST}" ]; then
    echo "ERROR: Phase 1 best model not found at ${PHASE1_BEST}"
    echo "Run Phase 1 first: sbatch --export=MODEL=${MODEL} slurm/phase1_train.sbatch"
    exit 1
fi

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs"

# ---- Info ----
echo "=== Phase 2: Full fine-tuning ==="
echo "Model: ${MODEL}"
echo "GPUs: ${NGPUS}"
echo "Grad accum steps: ${GRAD_ACCUM}"
echo "Phase 1 checkpoint: ${PHASE1_BEST}"
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Train ----
if [ "${NGPUS}" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=${NGPUS} --master_port=$((29500 + RANDOM % 1000))"
else
    CMD="python"
fi

LAST_CKPT="${OUTPUT_DIR}/vit_${MODEL}_${EXP_SUFFIX}/last.pth.tar"
if [ -f "${LAST_CKPT}" ]; then
    echo "Resuming from ${LAST_CKPT}"
    CHECKPOINT_ARG="--resume ${LAST_CKPT}"
else
    echo "Starting from Phase 1 checkpoint: ${PHASE1_BEST}"
    CHECKPOINT_ARG="--initial-checkpoint ${PHASE1_BEST}"
    # Load EMA head weights from Phase 1 (much better than instantaneous weights
    # when P1 stopped early with high LR). Default: on. Disable with USE_EMA_INIT=0.
    if [ "${USE_EMA_INIT}" = "1" ]; then
        CHECKPOINT_ARG="${CHECKPOINT_ARG} --initial-checkpoint-use-ema"
        echo "Will load EMA weights (state_dict_ema) from Phase 1 checkpoint"
    else
        echo "Will load regular weights (state_dict) from Phase 1 checkpoint"
    fi
fi

${CMD} "${EXP_DIR}/train_vit.py" \
    -c "${EXP_DIR}/configs/phase2_${MODEL}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    ${CHECKPOINT_ARG} \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --output "${OUTPUT_DIR}" \
    --experiment "vit_${MODEL}_${EXP_SUFFIX}"

echo "=== Phase 2 complete: ${MODEL} ==="
echo "Best model: ${OUTPUT_DIR}/vit_${MODEL}_${EXP_SUFFIX}/model_best.pth.tar"
