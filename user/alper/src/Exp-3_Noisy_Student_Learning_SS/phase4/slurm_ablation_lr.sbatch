#!/bin/bash
#SBATCH --job-name=p4-abl-lr
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Ablation: Learning rate (3 runs)
#   0: lr=3e-5
#   1: lr=1e-4 (baseline)
#   2: lr=3e-4

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

DATASET="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase3_student_dataset/lucas_pseudo_labels.parquet"
BASE_OUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/phase4_checkpoints"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase4/ablation_sweep.py \
  --category lr \
  --run-idx "${SLURM_ARRAY_TASK_ID}" \
  --dataset "${DATASET}" \
  --base-out-dir "${BASE_OUT_DIR}"

echo "Done: ablation_lr run ${SLURM_ARRAY_TASK_ID}"
