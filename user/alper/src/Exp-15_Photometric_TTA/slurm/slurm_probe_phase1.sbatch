#!/bin/bash
#SBATCH --job-name=exp15-probe-phase1
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Phase 1: Probe ViT-B+LoRA on the Exp-15 stratified probe sample, applying
# each of {none, unsharp, clahe_lab, saturation} photometric variants per tile.
# Output: outputs/phase1/probe_logits.parquet (used by score_phase1.py).

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

EXP15_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-15_Photometric_TTA"
EXP9B_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"
export PYTHONPATH="${EXP9B_ROOT}:${EXP15_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs outputs/phase1

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi || true

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
BACKBONE_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_lora/lora_head_final.pt"

PROBE_CSV="${EXP15_ROOT}/outputs/probe/probe_sample.csv"
OUT="${EXP15_ROOT}/outputs/phase1/probe_logits.parquet"

if [[ ! -f "$PROBE_CSV" ]]; then
    echo "Probe CSV not found at $PROBE_CSV — run scripts/build_probe_sample.py first."
    exit 1
fi

echo "=== Exp-15 Phase 1 probe ==="
echo "Image dir:   $IMAGE_DIR"
echo "Probe CSV:   $PROBE_CSV"
echo "Out:         $OUT"

cd "$EXP15_ROOT"
python scripts/probe_vit_variants.py \
    --probe-csv "$PROBE_CSV" \
    --image-dir "$IMAGE_DIR" \
    --backbone-path "$BACKBONE_PATH" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    --scales "4,5,6" \
    --batch-size 32

echo ""
echo "=== Phase 1 probe extraction done ==="
echo "Next: run scripts/score_phase1.py locally on this parquet."
