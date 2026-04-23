#!/bin/bash
#SBATCH --job-name=exp9b-project-emb
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-9B: apply trained ArcFace head to cached 768-d CLS embeddings.
# Runs train + test in sequence. Head-only forward passes, so this is mostly
# matmul + parquet I/O. Expected runtime: ~15-30 min total.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-9B_Improved_KNN_Retrieval"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
nvidia-smi || true

CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/arcface_head/arcface_head_final.pt"
TRAIN_IN="${PLANTCLEF_ROOT}/embeddings/train"
TRAIN_OUT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/train"

# Test tile parquets produced by Exp-7 — scale_4x4.parquet, scale_5x5.parquet,
# scale_6x6.parquet live directly under this directory (flat layout).
TEST_IN="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
TEST_OUT="${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/test"

echo "=== Exp-9B: project TRAIN embeddings ==="
python "${SRC_ROOT}/project_embeddings.py" \
  --mode train \
  --checkpoint "${CHECKPOINT}" \
  --in-dir "${TRAIN_IN}" \
  --out-dir "${TRAIN_OUT}" \
  --batch-size 32768

echo ""
echo "=== Exp-9B: project TEST tile embeddings ==="
python "${SRC_ROOT}/project_embeddings.py" \
  --mode test \
  --checkpoint "${CHECKPOINT}" \
  --in-dir "${TEST_IN}" \
  --out-dir "${TEST_OUT}" \
  --batch-size 32768 \
  --emb-col-in embedding_orig_f16 \
  --emb-col-out emb_arcface_f16

echo ""
echo "=== Projection done ==="
echo "Train out: ${TRAIN_OUT}"
echo "Test out:  ${TEST_OUT}"
