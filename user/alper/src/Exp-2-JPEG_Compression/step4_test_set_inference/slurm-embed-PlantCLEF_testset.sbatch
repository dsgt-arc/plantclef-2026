#!/bin/bash
#SBATCH --job-name=pc26-test-emb
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=embers
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

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
export MODEL_PATH

mkdir -p logs

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}  ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

# Helps reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate your venv
cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

DATASET_NAME="plantclef_2026_test"

IMAGES_DIR="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/PlanCLEF_test_data/images"

MANIFEST_BASE="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/manifests/${DATASET_NAME}/sharded_4"

NUM_MANIFEST_SHARDS=4
MANIFEST_SHARD_ID="${SLURM_ARRAY_TASK_ID}"

if [ "${MANIFEST_SHARD_ID}" -lt 0 ] || [ "${MANIFEST_SHARD_ID}" -ge "${NUM_MANIFEST_SHARDS}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${MANIFEST_SHARD_ID} is out of range 0..$((NUM_MANIFEST_SHARDS-1))"
  exit 2
fi

MANIFEST_PATH="${MANIFEST_BASE}/test_manifest_shard_$(printf "%02d" ${MANIFEST_SHARD_ID}).txt"

OUTPUT_ROOT="${PLANTCLEF_ROOT}/parquet/${DATASET_NAME}/feature_bank_JPEG_compressed"
OUT_SHARD="${OUTPUT_ROOT}/shard_$(printf "%04d" ${MANIFEST_SHARD_ID})"

mkdir -p "${OUTPUT_ROOT}"

if [ ! -d "${IMAGES_DIR}" ]; then
  echo "ERROR: IMAGES_DIR does not exist: ${IMAGES_DIR}"
  exit 2
fi

if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "ERROR: Manifest shard does not exist: ${MANIFEST_PATH}"
  exit 2
fi

echo "Using manifest: ${MANIFEST_PATH}"
echo "Writing output shard: ${OUT_SHARD}"

echo "Manifest count:"
wc -l "${MANIFEST_PATH}" || true
echo "First 5 lines:"
head -n 5 "${MANIFEST_PATH}" || true

rm -rf "${OUT_SHARD}"

TASK_NAME="${DATASET_NAME}_embed_shard$(printf "%04d" ${MANIFEST_SHARD_ID})"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step4_test_set_inference/extract_testdata_embeddings.py ExtractLUCAS_Embeddings \
  --dataset-name "${DATASET_NAME}" \
  --input-path "${IMAGES_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-path "${OUT_SHARD}" \
  --model-path "${MODEL_PATH}" \
  --task "${TASK_NAME}" \
  --manifest-path "${MANIFEST_PATH}" \
  --shard-id "${MANIFEST_SHARD_ID}" \
  --num-shards 1 \
  --cpu-count "${SLURM_CPUS_PER_TASK}" \
  --num-partitions 1 \
  --batch-size 16 \
  --grid-size 4 \
  --jpeg-quality 85 \
  --jpeg-subsampling 1 \
  --single-file \
  --write-image-name \
  --overwrite \
  --local-scheduler

ERR_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
if [ -f "${ERR_LOG}" ] && grep -q "This progress looks :(" "${ERR_LOG}"; then
  echo "Luigi reported failed tasks in ${ERR_LOG}"
  exit 1
fi

echo "Done. Output shard: ${OUT_SHARD}"
ls -lah "${OUT_SHARD}" || true