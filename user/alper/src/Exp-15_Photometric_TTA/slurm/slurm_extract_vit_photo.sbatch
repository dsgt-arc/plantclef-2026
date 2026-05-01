#!/bin/bash
#SBATCH --job-name=exp15-extract-vit-photo
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Phase 2: full ViT test re-extraction with one photometric variant.
# Set the variant via the PHOTO env var before sbatch:
#   PHOTO=unsharp    sbatch slurm_extract_vit_photo.sbatch
#   PHOTO=clahe_lab  sbatch slurm_extract_vit_photo.sbatch
#   PHOTO=saturation sbatch slurm_extract_vit_photo.sbatch

set -euo pipefail
export PYTHONUNBUFFERED=1

PHOTO="${PHOTO:?Set PHOTO env var to one of unsharp|clahe_lab|saturation}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

EXP15_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-15_Photometric_TTA"
EXP9B_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
export PYTHONPATH="${EXP9B_ROOT}:${EXP15_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}  PHOTO=${PHOTO}"
nvidia-smi || true

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
BACKBONE_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_lora/lora_head_final.pt"
EXISTING="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface_lora/test"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp15_vit_lora_${PHOTO}"

echo "=== Exp-15 Phase 2: ViT photometric extraction (${PHOTO}) ==="
echo "Image dir:   $IMAGE_DIR"
echo "Existing:    $EXISTING"
echo "Out:         $OUT_DIR"

cd "$EXP15_ROOT"
python scripts/extract_embeddings_lora_test_photo.py \
    --image-dir "$IMAGE_DIR" \
    --backbone-path "$BACKBONE_PATH" \
    --checkpoint "$CHECKPOINT" \
    --existing-parquet-dir "$EXISTING" \
    --out-dir "$OUT_DIR" \
    --scales "4,5,6" \
    --batch-size 64 \
    --jpeg-quality 85 \
    --photometric "$PHOTO"

echo ""
echo "=== Phase 2 extraction done (${PHOTO}) ==="
echo "Next steps:"
echo "  1. Run multiscale_aggregate_knn.py on \"$OUT_DIR\" to produce a"
echo "     knn_bM0.85_bS1.00_T1.50_prefusion.parquet under the photometric variant."
echo "  2. Run scripts/blend_vit_photo_tta.py to sweep beta, then ensemble_v2.py"
echo "     + apply_habitat_mask.py for the final submission CSVs."
