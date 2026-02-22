#!/bin/bash
#SBATCH --job-name=pc26-test-emb
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail
: "${PLANTCLEF_ROOT:?PLANTCLEF_ROOT is not set}"

mkdir -p logs

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

# Helps reduce CUDA memory fragmentation in long-running jobs
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate your venv
cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

# Paths
DATASET_NAME="plantclef_2026_test"
IMAGES_DIR="/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/PlanCLEF_test_data/images"

MANIFEST_DIR="${PLANTCLEF_ROOT}/manifests/${DATASET_NAME}"
MANIFEST_PATH="${MANIFEST_DIR}/test_manifest.txt"

# Output root for the embedding parquet
OUTPUT_ROOT="${PLANTCLEF_ROOT}/parquet/${DATASET_NAME}/feature_bank_single"
OUT_SHARD="${OUTPUT_ROOT}/shard_0000"

mkdir -p "${MANIFEST_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# Create manifest if missing
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "Creating manifest at: ${MANIFEST_PATH}"
  find "${IMAGES_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) > "${MANIFEST_PATH}"
fi

echo "Manifest count:"
wc -l "${MANIFEST_PATH}" || true
echo "First 5 lines:"
head -n 5 "${MANIFEST_PATH}" || true

# During debugging, it's safer to remove partial outputs from previous failed attempts
rm -rf "${OUT_SHARD}"

# Run extractor
python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step1/extract_lucas_embeddings.py ExtractLUCAS_Embeddings \
  --task embeddings \
  --dataset-name "${DATASET_NAME}" \
  --input-path "${IMAGES_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-path "${OUT_SHARD}" \
  --manifest-path "${MANIFEST_PATH}" \
  --shard-id 0 \
  --num-shards 1 \
  --cpu-count "${SLURM_CPUS_PER_TASK}" \
  --num-partitions 20 \
  --batch-size 32 \
  --grid-size 4 \
  --single-file True \
  --overwrite True \
  --local-scheduler

# Luigi can sometimes report failed tasks without a nonzero exit code.
# Force FAIL if Luigi indicates failures.
OUT_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
if grep -q "This progress looks :(" "${OUT_LOG}"; then
  echo "Luigi reported failed tasks in ${OUT_LOG}"
  exit 1
fi

echo "Done. Output shard: ${OUT_SHARD}"
ls -lah "${OUT_SHARD}" || true


