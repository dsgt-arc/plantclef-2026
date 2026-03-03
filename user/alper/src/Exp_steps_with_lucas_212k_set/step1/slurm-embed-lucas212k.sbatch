#!/bin/bash
#SBATCH --job-name=lucas212k-emb
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-212%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail
: "${PLANTCLEF_ROOT:?PLANTCLEF_ROOT is not set}"

mkdir -p logs

cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  ArrayJobID: ${SLURM_ARRAY_JOB_ID:-}  TaskID: ${SLURM_ARRAY_TASK_ID:-}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

DATASET_NAME="lucas_212k"
INPUT_PATH="${PLANTCLEF_ROOT}/data/raw/lucas_212k_images"
OUTPUT_ROOT="${PLANTCLEF_ROOT}/parquet/${DATASET_NAME}/feature_bank"
MANIFEST_DIR="${PLANTCLEF_ROOT}/manifests/${DATASET_NAME}"

# Helps reduce CUDA memory fragmentation in long-running jobs
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Shard-specific output directory to avoid any accidental overwrite
OUT_SHARD="${OUTPUT_ROOT}/shard_$(printf '%04d' ${SLURM_ARRAY_TASK_ID})"

# During debugging, it's safer to remove partial outputs from previous failed attempts
rm -rf "${OUT_SHARD}"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step1/extract_lucas_embeddings.py ExtractLUCAS_Embeddings \
  --task embeddings \
  --dataset-name "${DATASET_NAME}" \
  --input-path "${INPUT_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-path "${OUT_SHARD}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 213 \
  --cpu-count "${SLURM_CPUS_PER_TASK}" \
  --num-partitions 1 \
  --batch-size 8 \
  --grid-size 4 \
  --local-scheduler

# Luigi can sometimes report failed tasks without a nonzero exit code.
# Force the SLURM array element to FAIL if Luigi indicates failures.
OUT_LOG="logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
if grep -q "This progress looks :(" "${OUT_LOG}"; then
  echo "Luigi reported failed tasks in ${OUT_LOG}"
  exit 1
fi

echo "Done. Output shard: ${OUT_SHARD}"
