#!/bin/bash
# Exp-11C smoke test: validate the LoRA+ArcFace+Mixup pipeline on RTX 6000
# before committing 18h on H200.
#
# Verifies:
#   1. phase2v3 ViT-L .pth.tar loads (state_dict_ema path)
#   2. LoRA injects on 24 blocks (vs ViT-B's 12)
#   3. Mixup/CutMix wrapper produces soft targets
#   4. SubCenterArcFace.forward handles 2-D soft targets
#   5. soft_cross_entropy loss is finite, not NaN
#   6. End-of-epoch checkpoint saves cleanly
#   7. Memory fits at batch=4 on 22 GB RTX 6000 with grad checkpointing
#
# Caps dataset to max-per-class=2 → ~15K samples → ~3900 steps at batch=4
# → ~45 min wallclock on RTX 6000.

#SBATCH --job-name=exp11c-vitL-lora-smoke
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

DATA_DIR="${PLANTCLEF_ROOT}/data/raw/train_full"
BACKBONE_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit/vit_dinov2_large_phase2v3/model_best.pth.tar"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface_smoketest"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} missing"
    exit 1
fi
if [ ! -f "${BACKBONE_PATH}" ]; then
    echo "ERROR: phase2v3 backbone not found at ${BACKBONE_PATH}"
    exit 1
fi

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Exp-11C SMOKE TEST: LoRA+ArcFace+Mixup on phase2v3 ViT-L ==="
echo "Note:    --max-per-class 2 caps dataset to ~15K samples (~45 min on RTX 6000)"
echo "Data:    ${DATA_DIR}"
echo "Out:     ${OUT_DIR}"

python "${SRC_ROOT}/train_arcface_lora_vitL.py" \
    --data-dir "${DATA_DIR}" \
    --backbone-path "${BACKBONE_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --out-dir "${OUT_DIR}" \
    --epochs 1 \
    --batch-size 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --warmup-steps 50 \
    --margin 0.30 \
    --scale 30.0 \
    --k-sub 3 \
    --out-dim 1024 \
    --label-smoothing 0.10 \
    --image-size 518 \
    --jpeg-quality 85 \
    --lora-r 8 \
    --lora-alpha 16.0 \
    --lora-dropout 0.0 \
    --num-workers 4 \
    --log-every 10 \
    --max-per-class 2 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup-prob 1.0 \
    --seed 42

echo ""
echo "=== Smoke test done ==="
echo "If train loss decreased and lora_head_ep1.pt saved, the H200 run will work."
echo "Expected: loss starts ~10 (random init), drops to ~6-8 by step 1000 (margin warmup)."
