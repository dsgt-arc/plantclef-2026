#!/bin/bash
# Exp-11C Step 2: LoRA+ArcFace+Mixup/CutMix on phase2v3 ViT-L.
#
# Backbone: vit_dinov2_large_phase2v3/model_best.pth.tar (frozen + LoRA on attn.qkv)
# LoRA:     rank=8, alpha=16, all 24 attention blocks
# Head:     1024 -> 1024 -> BN -> ReLU -> 1024 -> L2-norm
# Arc:      sub-center K=3, m=0.30, s=30
# Aug:      Mixup α=0.8, CutMix α=1.0, prob=1.0
# Data:     train_full (train + val + test combined ImageFolder)
# Expected: ~5h28m (Exp-9B ViT-B, 1 H200) × 2.7 ≈ 15h on 1 H200
#
# Usage:
#   sbatch slurm/train_lora_vitL.sbatch

#SBATCH --job-name=exp11c-vitL-lora
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} missing — run link_dataset.sbatch first"
    exit 1
fi
if [ ! -f "${BACKBONE_PATH}" ]; then
    echo "ERROR: phase2v3 backbone not found at ${BACKBONE_PATH}"
    exit 1
fi

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Exp-11C Step 2: LoRA+ArcFace+Mixup/CutMix on phase2v3 ViT-L ==="
echo "Data:       ${DATA_DIR}"
echo "Backbone:   ${BACKBONE_PATH}"
echo "Output:     ${OUT_DIR}"

python "${SRC_ROOT}/train_arcface_lora_vitL.py" \
    --data-dir "${DATA_DIR}" \
    --backbone-path "${BACKBONE_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --out-dir "${OUT_DIR}" \
    --epochs 3 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --warmup-steps 500 \
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
    --num-workers 6 \
    --log-every 50 \
    --max-per-class 0 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup-prob 1.0 \
    --seed 42

echo ""
echo "=== Done. Final checkpoint: ${OUT_DIR}/lora_head_final.pt ==="
echo "Next: extract embeddings via project_embeddings_vitL.sbatch (port from Exp-9B),"
echo "then build_faiss_index_vitL.sbatch."
