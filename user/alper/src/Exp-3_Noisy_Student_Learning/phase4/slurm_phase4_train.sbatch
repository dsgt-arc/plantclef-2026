#!/bin/bash
#SBATCH --job-name=p4-student-train
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

DATASET="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase3_student_dataset/lucas_pseudo_labels.parquet"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/phase4_checkpoints/run_kl_default"

# UPDATE partition name if needed (check: scontrol show partition)
# Common PACE GPU partitions: gpu-v100, gpu-a100, gpu-h100

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase4/train_student.py \
  --dataset "${DATASET}" \
  --out-dir "${OUT_DIR}" \
  --loss-type kl \
  --cons-mode logit_mse \
  --lambda-cons 0.01 \
  --warmup-epochs 5 \
  --min-image-conf 0.0 \
  --loss-topk 0 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --tile-strategy none \
  --hidden-dim 512 \
  --num-heads 8 \
  --num-layers 4 \
  --dropout 0.1 \
  --embed-dropout 0.1 \
  --amp \
  --save-every 5 \
  --seed 42

echo "Done: Phase 4 student training"
