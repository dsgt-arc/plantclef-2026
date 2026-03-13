#!/bin/bash
#SBATCH --job-name=exp4-test-jpeg85
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

EXP3_SRC="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning"
export PYTHONPATH="${EXP3_SRC}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayJobID: ${SLURM_ARRAY_JOB_ID:-}  TaskID: ${SLURM_ARRAY_TASK_ID:-}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi || true

# Re-extract test embeddings with JPEG compression (quality=85, 4:2:2 subsampling)
# Replicates Exp-2 finding: JPEG round-trip aligns pixel stats with DINOv2 training data
# Exp-2 result: +0.038 F1 from JPEG compression alone
MANIFEST_DIR="${PLANTCLEF_ROOT}/aerten3/manifests/plantclef_2026_test/new_shards"
OUTPUT_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

OUT_SHARD="${OUTPUT_ROOT}/shard_$(printf '%04d' ${SLURM_ARRAY_TASK_ID})"
rm -rf "${OUT_SHARD}"

echo "=== Extracting test embeddings with JPEG compression ==="
echo "JPEG quality: 85, subsampling: 1 (4:2:2)"
echo "Output: ${OUT_SHARD}"

python "${EXP3_SRC}/phase1/extract_embeddings.py" ExtractPhase1Embeddings \
  --dataset-type test \
  --input-path "${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images" \
  --checkpoint-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --cpu-count "${SLURM_CPUS_PER_TASK}" \
  --num-partitions 1 \
  --batch-size 8 \
  --jpeg-quality 85 \
  --jpeg-subsampling 1 \
  --overwrite \
  --local-scheduler

OUT_LOG="logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
if grep -q "This progress looks :(" "${OUT_LOG}" 2>/dev/null; then
  echo "Luigi reported failed tasks"
  exit 1
fi

echo "Done. Output shard: ${OUT_SHARD}"
