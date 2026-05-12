#!/bin/bash
# Exp-11C add-on: extract scale 3x3 ArcFace tile embeddings on PACE.
#
# The main extraction (extract_lora_test_vitL.sbatch) ran for scales 4,5,6,7.
# This add-on does just scale 3 — needed for the Abl-4 "scale 3+6" finding
# (which beat the production 4+5+6+7 by +0.01676 priv F1).
#
# Output merges with the existing phase2v3 scale_3x3.parquet so the result
# has both teacher_logits_f16 (classifier) and emb_arcface_f16 (LoRA retriever).
#
# Cost: ~30-60 min on H200 (9 tiles/image, 18,945 tiles total).
#
# Usage:
#   sbatch slurm/extract_lora_test_vitL_scale3.sbatch

#SBATCH --job-name=exp11c-extract-lora-scale3
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

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
BACKBONE_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit/vit_dinov2_large_phase2v3/model_best.pth.tar"
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/lora_head_final.pt"
EXISTING="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_dinov2_large_phase2v3"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/embeddings/test"

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "ERROR: ${IMAGE_DIR} missing"
    exit 1
fi
if [ ! -f "${BACKBONE_PATH}" ]; then
    echo "ERROR: phase2v3 backbone not found at ${BACKBONE_PATH}"
    exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: LoRA checkpoint not found at ${CHECKPOINT}"
    exit 1
fi
if [ ! -f "${EXISTING}/scale_3x3.parquet" ]; then
    echo "ERROR: phase2v3 scale_3x3 parquet missing at ${EXISTING}/scale_3x3.parquet"
    echo "This script merges with it. Run extract_multiscale_3to7.sbatch with EXP_SUFFIX=phase2v3 first."
    exit 1
fi

# Don't clobber a prior run if it's already there
if [ -f "${OUT_DIR}/scale_3x3.parquet" ]; then
    echo "NOTE: ${OUT_DIR}/scale_3x3.parquet already exists. It will be overwritten."
fi

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Exp-11C scale 3x3 LoRA extraction ==="
echo "Image dir:           ${IMAGE_DIR}"
echo "Existing classifier: ${EXISTING}/scale_3x3.parquet"
echo "LoRA checkpoint:     ${CHECKPOINT}"
echo "Out:                 ${OUT_DIR}/scale_3x3.parquet"

python "${SRC_ROOT}/extract_embeddings_lora_test_vitL.py" \
  --image-dir "${IMAGE_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --existing-parquet-dir "${EXISTING}" \
  --out-dir "${OUT_DIR}" \
  --scales "3" \
  --batch-size 32 \
  --jpeg-quality 85

echo ""
echo "=== Done. ${OUT_DIR}/scale_3x3.parquet has both ==="
echo "  - teacher_logits_f16  (phase2v3 classifier output, inherited)"
echo "  - emb_arcface_f16     (1024-d LoRA+ArcFace retriever embedding, new)"
echo ""
echo "Next: download this single parquet to your laptop and rebuild the"
echo "kNN prefusion at scales 3+6 locally with Exp-14C/make_vitL_prefusion.py."
