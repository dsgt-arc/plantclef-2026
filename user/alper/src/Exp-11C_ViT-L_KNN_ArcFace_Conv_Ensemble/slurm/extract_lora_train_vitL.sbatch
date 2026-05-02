#!/bin/bash
# Exp-11C: project ~1.4M PlantCLEF train images through phase2v3 ViT-L
# + LoRA + ArcFace head -> 1024-d normalized embeddings.
# Writes sample_id=*/ shards consumable by build_faiss_index_vitL.sbatch.
#
# Usage:
#   sbatch slurm/extract_lora_train_vitL.sbatch

#SBATCH --job-name=exp11c-extract-lora-train
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

DATA_DIR="${PLANTCLEF_ROOT}/data/raw/train_full"
BACKBONE_PATH="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vit/vit_dinov2_large_phase2v3/model_best.pth.tar"
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/lora_head_final.pt"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp11c_vitL_lora_arcface/embeddings/train"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} missing"
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

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Exp-11C: project train embeddings ==="
echo "Data:       ${DATA_DIR}"
echo "Backbone:   ${BACKBONE_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Out:        ${OUT_DIR}"

python "${SRC_ROOT}/extract_embeddings_lora_train_vitL.py" \
  --data-dir "${DATA_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --species-map "${SPECIES_MAP}" \
  --out-dir "${OUT_DIR}" \
  --batch-size 96 \
  --num-workers 8 \
  --shard-size 70000 \
  --jpeg-quality 0

echo ""
echo "=== Train extraction done ==="
echo "Next: sbatch slurm/build_faiss_index_vitL.sbatch"
