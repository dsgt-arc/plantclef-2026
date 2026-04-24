#!/bin/bash
#SBATCH --job-name=exp9b-train-arcface
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B Tier 1: train ArcFace projection head on cached DINOv2 CLS embeddings.
# Head-only — backbone stays frozen. Trains on ~1.4M cached 768-d vectors.
#
# Expected runtime: ~30-90 minutes on one RTX 6000 at batch 16384, 5 epochs.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi || true

TRAIN_EMB_DIR="${PLANTCLEF_ROOT}/embeddings/train"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_head"

echo "=== Exp-9B: ArcFace head training ==="
echo "Train embeddings: ${TRAIN_EMB_DIR}"
echo "Checkpoint out:   ${OUT_DIR}"

python "${SRC_ROOT}/train_arcface_head.py" \
  --train-emb-dir "${TRAIN_EMB_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --out-dir "${OUT_DIR}" \
  --epochs 15 \
  --batch-size 16384 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --warmup-steps 100 \
  --margin 0.30 \
  --scale 30.0 \
  --k-sub 3 \
  --out-dim 768 \
  --label-smoothing 0.10 \
  --num-workers 4 \
  --log-every 20 \
  --seed 42

echo ""
echo "=== Training done ==="
echo "Final checkpoint: ${OUT_DIR}/arcface_head_final.pt"
