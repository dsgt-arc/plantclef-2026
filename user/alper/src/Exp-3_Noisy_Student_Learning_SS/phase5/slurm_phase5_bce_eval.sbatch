#!/bin/bash
#SBATCH --job-name=p5-loss-eval
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Phase 5 inference on loss (neg sampling) ablation checkpoints (4 runs)
#   0: bce_neg256_baseline
#   1: bce_neg64
#   2: bce_neg128
#   3: bce_neg512

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

# Map array task ID to run name
RUN_NAMES=(bce_neg256_baseline bce_neg64 bce_neg128 bce_neg512)
RUN_NAME="${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}"

CKPT_BASE="${PLANTCLEF_ROOT}/aerten3/checkpoints/phase4_checkpoints/ablation_loss"
CHECKPOINT="${CKPT_BASE}/${RUN_NAME}/best_model.pt"

TILE_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/phase5_loss_eval/${RUN_NAME}"
OUT_PARQUET="${OUT_DIR}/student_probs.parquet"
OUT_CSV="${OUT_DIR}/student_submission_${RUN_NAME}.csv"

echo "Evaluating: ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUT_DIR}"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase5/student_inference.py \
  --tile-parquet-root "${TILE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --out-parquet "${OUT_PARQUET}" \
  --out-csv "${OUT_CSV}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --topk 10 \
  --threshold 0.03 \
  --aug-filter \
  --tile-filter-k 16 \
  --batch-size 64

echo "Done: Phase 5 eval for ${RUN_NAME}"
