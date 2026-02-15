#!/bin/bash
#SBATCH --job-name=lucas212k-emb
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu
#SBATCH --qos=embers
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail
: "${PLANTCLEF_ROOT:?PLANTCLEF_ROOT is not set}"

mkdir -p logs

# Activate YOUR working environment
cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction
source .venv/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvidia-smi

export DATASET_NAME=lucas_212k
export INPUT_PATH=$PLANTCLEF_ROOT/data/raw/lucas_212k_images
export OUTPUT_ROOT=$PLANTCLEF_ROOT/parquet/${DATASET_NAME}/feature_bank
export MANIFEST_DIR=$PLANTCLEF_ROOT/manifests/${DATASET_NAME}

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step1/extract_lucas_embeddings.py ExtractLUCAS_Embeddings \
  --dataset-name ${DATASET_NAME} \
  --input-path ${INPUT_PATH} \
  --output-root ${OUTPUT_ROOT} \
  --manifest-dir ${MANIFEST_DIR} \
  --shard-id ${SLURM_ARRAY_TASK_ID} \
  --num-shards 106 \
  --cpu-count ${SLURM_CPUS_PER_TASK} \
  --num-partitions 48 \
  --batch-size 16 \
  --grid-size 4
