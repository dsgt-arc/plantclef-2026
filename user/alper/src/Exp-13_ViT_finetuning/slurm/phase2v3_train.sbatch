#!/bin/bash
# Phase 2v3 — short cooldown extension on train+val+test combined data.
# Initialize from Phase 2v2's best EMA checkpoint, run 10 epochs at low LR.
# Uses train_full/ (symlinked combined dir, must be created via link_dataset.sbatch).
#
# Usage:
#   sbatch --export=MODEL=dinov2_large slurm/phase2v3_train.sbatch

#SBATCH --job-name=vit-p2v3
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase2v3_%x_%A.out
#SBATCH --error=logs/phase2v3_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

MODEL="${MODEL:-dinov2_large}"
NGPUS="${NGPUS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EXP_SUFFIX="${EXP_SUFFIX:-phase2v3}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-13_ViT_finetuning"
# Combined train+val+test symlinked dir lives under data/raw/, not train/PlantCLEF2024/
DATA_DIR="${PLANTCLEF_ROOT}/data/raw"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit"

# Source checkpoint: Phase 2v2 best EMA
SRC_CKPT="${SRC_CKPT:-${OUTPUT_DIR}/vit_${MODEL}_phase2v2/model_best.pth.tar}"

# Verify combined data dir exists
COMBINED_DIR="${DATA_DIR}/train_full"
if [ ! -d "${COMBINED_DIR}" ]; then
    echo "ERROR: combined train_full dir not found at ${COMBINED_DIR}"
    echo "Run link_dataset.sbatch first (output target: ${COMBINED_DIR})."
    exit 1
fi

if [ ! -f "${SRC_CKPT}" ]; then
    echo "ERROR: Phase 2v2 checkpoint not found at ${SRC_CKPT}"
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "${EXP_DIR}/logs"

echo "=== Phase 2v3 cooldown ==="
echo "Model:           ${MODEL}"
echo "GPUs:            ${NGPUS}"
echo "Source ckpt:     ${SRC_CKPT}"
echo "Combined data:   ${COMBINED_DIR}"
echo "Output suffix:   ${EXP_SUFFIX}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
    echo "Starting from Phase 2v2 EMA checkpoint"
    CHECKPOINT_ARG="--initial-checkpoint ${SRC_CKPT} --initial-checkpoint-use-ema"
fi

${CMD} "${EXP_DIR}/train_vit.py" \
    -c "${EXP_DIR}/configs/phase2v3_${MODEL}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train_full \
    --val-split train_full \
    ${CHECKPOINT_ARG} \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --output "${OUTPUT_DIR}" \
    --experiment "vit_${MODEL}_${EXP_SUFFIX}"

echo "=== Phase 2v3 complete: ${MODEL} ==="
echo "Best model: ${OUTPUT_DIR}/vit_${MODEL}_${EXP_SUFFIX}/model_best.pth.tar"
