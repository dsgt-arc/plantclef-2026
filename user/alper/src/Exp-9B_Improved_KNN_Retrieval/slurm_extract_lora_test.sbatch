#!/bin/bash
#SBATCH --job-name=exp9b-extract-lora-test
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B Tier 2: re-tile test images and extract new 768-d embeddings from the
# LoRA backbone + head. Merges onto existing Exp-7 parquets to preserve
# teacher_logits_f16 and companion columns.

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

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
BACKBONE_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_lora/lora_head_final.pt"
EXISTING="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface_lora/test"

echo "=== Exp-9B Tier 2: project test tile embeddings ==="
echo "Image dir:          ${IMAGE_DIR}"
echo "Existing parquets:  ${EXISTING}"
echo "Checkpoint:         ${CHECKPOINT}"
echo "Out:                ${OUT_DIR}"

python "${SRC_ROOT}/extract_embeddings_lora_test.py" \
  --image-dir "${IMAGE_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --existing-parquet-dir "${EXISTING}" \
  --out-dir "${OUT_DIR}" \
  --scales "4,5,6" \
  --batch-size 64 \
  --jpeg-quality 85

echo ""
echo "=== Test extraction done ==="
