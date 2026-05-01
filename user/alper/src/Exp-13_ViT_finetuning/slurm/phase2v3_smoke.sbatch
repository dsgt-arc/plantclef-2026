#!/bin/bash
# Phase 2v3 smoke test on 1 RTX 6000 GPU.
# 1 epoch, 5000 samples, batch=16. Validates symlinked data, checkpoint
# loading, and training loop. ~10-15 min wallclock.

#SBATCH --job-name=vit-p2v3-smoke
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/p2v3_smoke_%A.out
#SBATCH --error=logs/p2v3_smoke_%A.err

set -euo pipefail

MODEL="${MODEL:-dinov2_large}"
PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-13_ViT_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/data/raw"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit"
SRC_CKPT="${OUTPUT_DIR}/vit_${MODEL}_phase2v2/model_best.pth.tar"

[ -d "${DATA_DIR}/train_full" ] || { echo "ERROR: ${DATA_DIR}/train_full missing"; exit 1; }
[ -f "${SRC_CKPT}" ] || { echo "ERROR: ${SRC_CKPT} missing"; exit 1; }

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "${EXP_DIR}/logs"

echo "=== Phase 2v3 SMOKE TEST (1 RTX 6000, 1 epoch, 5000 samples) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Single GPU, smaller batch for 24GB, override epochs/samples to limit run
python "${EXP_DIR}/train_vit.py" \
    -c "${EXP_DIR}/configs/phase2v3_${MODEL}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train_full \
    --val-split train_full \
    --initial-checkpoint "${SRC_CKPT}" \
    --initial-checkpoint-use-ema \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --workers 6 \
    --epochs 1 \
    --train-num-samples 4000 \
    --val-num-samples 400 \
    --log-interval 10 \
    --output "${OUTPUT_DIR}" \
    --experiment "vit_${MODEL}_phase2v3_smoke"

echo "=== Smoke test done. ==="
echo "If train loss decreased and 'Data:' was small, the H200 run will work."
