#!/bin/bash
# Exp-11C: re-extract test tiles through phase2v3 ViT-L + LoRA + ArcFace head,
# producing 1024-d emb_arcface_f16 column per tile. Merges with the existing
# phase2v3 multi-scale parquets so teacher_logits_f16 (classifier branch) is
# preserved alongside the new retriever embedding.
#
# Scales 4,5,6,7 = scheme K (the production scheme).
#
# Usage:
#   sbatch slurm/extract_lora_test_vitL.sbatch

#SBATCH --job-name=exp11c-extract-lora-test
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
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
    echo "Run train_lora_vitL.sbatch first."
    exit 1
fi
for s in 4 5 6 7; do
    f="${EXISTING}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: phase2v3 ${s}x${s} parquet missing at $f"
        echo "Run extract_multiscale_3to7.sbatch with --array=4-7 EXP_SUFFIX=phase2v3 first."
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Exp-11C: project test tile embeddings (scheme K) ==="
echo "Image dir:          ${IMAGE_DIR}"
echo "Existing parquets:  ${EXISTING}"
echo "Checkpoint:         ${CHECKPOINT}"
echo "Out:                ${OUT_DIR}"

python "${SRC_ROOT}/extract_embeddings_lora_test_vitL.py" \
  --image-dir "${IMAGE_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --existing-parquet-dir "${EXISTING}" \
  --out-dir "${OUT_DIR}" \
  --scales "4,5,6,7" \
  --batch-size 32 \
  --jpeg-quality 85

echo ""
echo "=== Test extraction done ==="
echo "Output parquets in ${OUT_DIR} now have:"
echo "  - teacher_logits_f16 (phase2v3 classifier, preserved)"
echo "  - emb_arcface_f16    (1024-d LoRA+ArcFace retriever, new)"
echo ""
echo "Next: ensemble sweep — ViT-L K + ConvNeXt K + new kNN. Update existing"
echo "Exp-14/16 ensemble scripts to point at these parquets and the new FAISS"
echo "index at \${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/faiss_index/."
