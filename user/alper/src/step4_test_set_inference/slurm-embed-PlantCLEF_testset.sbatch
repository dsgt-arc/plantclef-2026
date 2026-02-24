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
nvidia-smi

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

DATASET_NAME="plantclef_2026_test"

# Verify this path exists on your system. If it is a typo, fix PlanCLEF -> PlantCLEF.
IMAGES_DIR="/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/PlanCLEF_test_data/images"

MANIFEST_DIR="${PLANTCLEF_ROOT}/manifests/${DATASET_NAME}"
MANIFEST_PATH="${MANIFEST_DIR}/test_manifest.txt"

OUTPUT_ROOT="${PLANTCLEF_ROOT}/parquet/${DATASET_NAME}/feature_bank_single"
OUT_SHARD="${OUTPUT_ROOT}/shard_0000"

mkdir -p "${MANIFEST_DIR}"
mkdir -p "${OUTPUT_ROOT}"

if [ ! -d "${IMAGES_DIR}" ]; then
  echo "ERROR: IMAGES_DIR does not exist: ${IMAGES_DIR}"
  exit 2
fi

if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "Creating manifest at: ${MANIFEST_PATH}"
  find "${IMAGES_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) > "${MANIFEST_PATH}"
fi

echo "Manifest count:"
wc -l "${MANIFEST_PATH}" || true
echo "First 5 lines:"
head -n 5 "${MANIFEST_PATH}" || true

rm -rf "${OUT_SHARD}"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step4_test_set_inference/extract_testdata_embeddings.py ExtractLUCAS_Embeddings \
  --task "plantclef_2026_test_embed_shard0000" \
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
  --single-file \
  --write-image-name \
  --overwrite \
  --local-scheduler

OUT_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
if grep -q "This progress looks :(" "${OUT_LOG}"; then
  echo "Luigi reported failed tasks in ${OUT_LOG}"
  exit 1
fi

echo "Done. Output shard: ${OUT_SHARD}"
ls -lah "${OUT_SHARD}" || true