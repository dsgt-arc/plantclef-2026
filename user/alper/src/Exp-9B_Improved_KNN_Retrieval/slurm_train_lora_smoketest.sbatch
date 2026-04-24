#!/bin/bash
#SBATCH --job-name=exp9b-lora-smoke
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Tier 2 SMOKE TEST on RTX 6000.
#
# Purpose: validate the full training pipeline end-to-end before committing
# to the 8-hour H200 run. Uses --max-per-class 5 to cap the dataset to ~40K
# images (5 images per species x 7806 species) so one epoch completes in
# ~10-15 minutes on RTX 6000.
#
# What to watch in the log:
#   1. ImageFolder scan: should find 7806 subdirs + ~40K samples after cap
#   2. LoRA injection: should report 12 linears touched, ~0.34M trainable
#   3. Loss should drop from ~9.5 to something meaningfully lower by step 500
#   4. in-batch rank-1 should climb from ~0.35 (frozen baseline) to >0.40+
#   5. Final checkpoint should write without errors
#
# If all four pass, the H200 full run is safe to launch.
#
# Checkpoint path is intentionally different from the main run so we don't
# overwrite a good full-training checkpoint.

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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_lora_smoketest"

echo "=== Exp-9B Tier 2 SMOKE TEST ==="
echo "Data:       ${DATA_DIR}"
echo "Out:        ${OUT_DIR}"
echo "Note:       --max-per-class 5 caps dataset to ~40K samples"

python "${SRC_ROOT}/train_arcface_lora.py" \
  --data-dir "${DATA_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --out-dir "${OUT_DIR}" \
  --epochs 1 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --warmup-steps 50 \
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
  --log-every 10 \
  --max-per-class 5 \
  --seed 42

echo ""
echo "=== Smoke test done ==="
echo "Review ${OUT_DIR}/train_log.jsonl for loss/rank1/cosine trends."
