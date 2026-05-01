#!/bin/bash
#SBATCH --job-name=exp9b-train-lora
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B Tier 2: LoRA + ArcFace fine-tune of DINOv2 for retrieval.
#
# Backbone: DINOv2 ViT-Base (frozen)
# LoRA:     attn.qkv across all 12 blocks, rank=8
# Head:     768 -> 768 projection, L2-normalized
# Loss:     Sub-center ArcFace (K=3, m=0.3, s=30) over 7806 species
#
# Data: ~1.4M PlantCLEF 2024 train images at 518x518.
# Expected: ~4-8h on H200 at batch=64. On RTX 6000 falls back to 10-16h
# (consider --batch-size 32 and --no-grad-checkpoint off in that case).

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi || true

DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024/train"
BACKBONE_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_lora"

echo "=== Exp-9B Tier 2: LoRA+ArcFace training ==="
echo "Data:       ${DATA_DIR}"
echo "Backbone:   ${BACKBONE_PATH}"
echo "Checkpoint: ${OUT_DIR}"

python "${SRC_ROOT}/train_arcface_lora.py" \
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
  --out-dim 768 \
  --label-smoothing 0.10 \
  --image-size 518 \
  --jpeg-quality 0 \
  --lora-r 8 \
  --lora-alpha 16.0 \
  --lora-dropout 0.0 \
  --num-workers 6 \
  --log-every 50 \
  --max-per-class 0 \
  --seed 42

echo ""
echo "=== Tier 2 training done ==="
echo "Final checkpoint: ${OUT_DIR}/lora_head_final.pt"
