#!/bin/bash
# ---- GPU config: set GPU_TYPE env var to choose partition ----
# Usage:
#   sbatch --export=VARIANT=base                           slurm/phase2_train.sbatch  # default: 1x H200
#   sbatch --export=VARIANT=base,GPU_TYPE=h100,NGPUS=4     slurm/phase2_train.sbatch  # 4x H100
#   sbatch --export=VARIANT=base,GPU_TYPE=h200,NGPUS=2     slurm/phase2_train.sbatch  # 2x H200
#   sbatch --export=VARIANT=base,GPU_TYPE=a100,NGPUS=2     slurm/phase2_train.sbatch  # 2x A100

#SBATCH --job-name=cnxt-p2
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --qos=inferno
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/phase2_%x_%A.out
#SBATCH --error=logs/phase2_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# ---- Configuration ----
VARIANT="${VARIANT:-base}"
GPU_TYPE="${GPU_TYPE:-h200}"
NGPUS="${NGPUS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# ---- Resolve GPU partition and GRES ----
case "${GPU_TYPE}" in
    h200)
        PARTITION="gpu-h200"
        GRES="gpu:h200:${NGPUS}"
        ;;
    h100)
        PARTITION="gpu-h100"
        GRES="gpu:h100:${NGPUS}"
        ;;
    a100)
        PARTITION="gpu-a100"
        GRES="gpu:a100:${NGPUS}"
        ;;
    *)
        echo "Unknown GPU_TYPE: ${GPU_TYPE}. Use h200, h100, or a100."
        exit 1
        ;;
esac

# Re-submit ourselves with the correct partition/gres if not already set
# (SLURM doesn't support dynamic #SBATCH, so we use scontrol)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Running on partition: ${PARTITION}, GRES: ${GRES}"
else
    echo "Submitting with: --partition=${PARTITION} --gres=${GRES}"
fi

CPUS=$((NGPUS * 8))
MEM=$((NGPUS * 64))G

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext"

PHASE1_BEST="${OUTPUT_DIR}/convnext_${VARIANT}_phase1/model_best.pth.tar"

# ---- Validate Phase 1 checkpoint exists ----
if [ ! -f "${PHASE1_BEST}" ]; then
    echo "ERROR: Phase 1 best model not found at ${PHASE1_BEST}"
    echo "Run Phase 1 first: sbatch --export=VARIANT=${VARIANT} slurm/phase1_train.sbatch"
    exit 1
fi

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs"

# ---- Info ----
echo "=== Phase 2: Full fine-tuning ==="
echo "Variant: ${VARIANT}"
echo "GPU type: ${GPU_TYPE}"
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

${CMD} "${EXP_DIR}/train_convnext.py" \
    -c "${EXP_DIR}/configs/phase2_${VARIANT}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    --initial-checkpoint "${PHASE1_BEST}" \
    --grad-accum-steps "${GRAD_ACCUM}" \
    --output "${OUTPUT_DIR}" \
    --experiment "convnext_${VARIANT}_phase2"

echo "=== Phase 2 complete: ${VARIANT} ==="
echo "Best model: ${OUTPUT_DIR}/convnext_${VARIANT}_phase2/model_best.pth.tar"
