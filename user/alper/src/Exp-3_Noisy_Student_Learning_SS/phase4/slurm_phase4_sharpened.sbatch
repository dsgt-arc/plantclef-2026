#!/bin/bash
#SBATCH --job-name=p4-sharp-train
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Phase 4: Train student on sharpened pseudo-labels (4 configs)
#   0: thr005_topk64
#   1: thr007_topk64
#   2: thr003_topk30
#   3: thr005_topk30
#
# Uses combined OVAT winner config: BCE, neg128, cons_off, tile_thr005

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

RUN_NAMES=(thr005_topk64 thr007_topk64 thr003_topk30 thr005_topk30)
RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"

DATASET="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase3_sharpened/${RUN_NAME}/lucas_pseudo_labels.parquet"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/phase4_checkpoints/sharpened_${RUN_NAME}"

echo "Training on sharpened data: ${RUN_NAME}"
echo "Dataset: ${DATASET}"
echo "Output:  ${OUT_DIR}"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase4/train_student.py \
  --dataset "${DATASET}" \
  --out-dir "${OUT_DIR}" \
  --loss-type bce \
  --num-negatives 128 \
  --cons-mode logit_mse \
  --cons-augment dropout \
  --lambda-cons 0.0 \
  --warmup-epochs 5 \
  --min-image-conf 0.0 \
  --loss-topk 0 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --tile-strategy threshold \
  --tile-min-prob 0.05 \
  --hidden-dim 512 \
  --num-heads 8 \
  --num-layers 4 \
  --dropout 0.1 \
  --embed-dropout 0.1 \
  --amp \
  --save-every 5 \
  --seed 42

echo "Done: Phase 4 sharpened training (${RUN_NAME})"
