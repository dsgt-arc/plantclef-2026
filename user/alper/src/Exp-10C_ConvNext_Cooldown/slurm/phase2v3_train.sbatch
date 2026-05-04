#!/bin/bash
# Exp-10C — short cooldown extension on train+val+test combined data
# for ConvNeXt-Large. Initialize from Phase 2's best EMA checkpoint, run 10 epochs at low LR.
# Uses train_full/ (symlinked combined dir; created earlier for Exp-13 phase2v3).
# Mirrors the Exp-13 phase2v3 cooldown recipe.
#
# Usage:
#   sbatch slurm/phase2v3_train.sbatch
#
# Override the source checkpoint location if your Phase 2 run was on 4 GPUs:
#   sbatch --export=SRC_CKPT=/storage/.../convnext_large_phase2_4gpu/model_best.pth.tar slurm/phase2v3_train.sbatch

#SBATCH --job-name=cnxt-p2v3
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --output=logs/phase2v3_%x_%A.out
#SBATCH --error=logs/phase2v3_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

VARIANT="${VARIANT:-large}"
NGPUS="${NGPUS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EXP_SUFFIX="${EXP_SUFFIX:-phase2v3}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10C_ConvNext_Cooldown"
# Reuse Exp-10's training script (no new code needed)
TRAIN_SCRIPT="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning/train_convnext.py"
# train_full is the combined (train + val + test) symlinked ImageFolder
DATA_DIR="${PLANTCLEF_ROOT}/data/raw"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext"

# Source checkpoint — defaults to the standard Phase 2 path; override with
# SRC_CKPT env var if your production run was the 4gpu variant.
SRC_CKPT="${SRC_CKPT:-${OUTPUT_DIR}/convnext_${VARIANT}_phase2/model_best.pth.tar}"

# Verify combined data dir exists (created during Exp-13 phase2v3 setup)
COMBINED_DIR="${DATA_DIR}/train_full"
if [ ! -d "${COMBINED_DIR}" ]; then
    echo "ERROR: combined train_full dir not found at ${COMBINED_DIR}"
    echo "Run link_dataset.sbatch (from Exp-13 setup) first."
    exit 1
fi

if [ ! -f "${SRC_CKPT}" ]; then
    echo "ERROR: Phase 2 ConvNeXt checkpoint not found at ${SRC_CKPT}"
    echo "Override path with: sbatch --export=SRC_CKPT=/path/to/model_best.pth.tar slurm/phase2v3_train.sbatch"
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "${EXP_DIR}/logs"

echo "=== Exp-10C: ConvNeXt phase2v3 cooldown ==="
echo "Variant:         ${VARIANT}"
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

LAST_CKPT="${OUTPUT_DIR}/convnext_${VARIANT}_${EXP_SUFFIX}/last.pth.tar"
if [ -f "${LAST_CKPT}" ]; then
    echo "Resuming from ${LAST_CKPT}"
    CHECKPOINT_ARG="--resume ${LAST_CKPT}"
else
    echo "Starting from Phase 2 EMA checkpoint"
    CHECKPOINT_ARG="--initial-checkpoint ${SRC_CKPT} --initial-checkpoint-use-ema"
fi

${CMD} "${TRAIN_SCRIPT}" \
    -c "${EXP_DIR}/configs/phase2v3_${VARIANT}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train_full \
    --val-split train_full \
    ${CHECKPOINT_ARG} \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --output "${OUTPUT_DIR}" \
    --experiment "convnext_${VARIANT}_${EXP_SUFFIX}"

echo "=== Phase 2v3 complete: ConvNeXt ${VARIANT} ==="
echo "Production checkpoint: ${OUTPUT_DIR}/convnext_${VARIANT}_${EXP_SUFFIX}/last.pth.tar"
echo "(model_best.pth.tar selection is contaminated; prefer last.pth.tar)"
echo ""
echo "Next: re-extract test embeddings at scales 4-7 with the new checkpoint."
echo "Reuse Exp-10/slurm/extract_embeddings.sbatch with VARIANT=${VARIANT} but"
echo "point MODEL_PATH at convnext_${VARIANT}_${EXP_SUFFIX}/last.pth.tar."
