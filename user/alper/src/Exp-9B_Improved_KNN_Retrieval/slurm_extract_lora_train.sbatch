#!/bin/bash
#SBATCH --job-name=exp9b-extract-lora-train
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B Tier 2: project ~1.4M PlantCLEF 2024 train images through the
# LoRA-adapted backbone + ArcFace head, to 768-d normalized embeddings.
#
# Writes sample_id=* parquet shards consumable by build_faiss_index_512.py.

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
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_lora/lora_head_final.pt"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface_lora/train"

echo "=== Exp-9B Tier 2: project train embeddings ==="
echo "Data:       ${DATA_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Out:        ${OUT_DIR}"

python "${SRC_ROOT}/extract_embeddings_lora_train.py" \
  --data-dir "${DATA_DIR}" \
  --backbone-path "${BACKBONE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --species-map "${SPECIES_MAP}" \
  --out-dir "${OUT_DIR}" \
  --batch-size 96 \
  --num-workers 6 \
  --shard-size 70000 \
  --jpeg-quality 0

echo ""
echo "=== Train extraction done ==="
