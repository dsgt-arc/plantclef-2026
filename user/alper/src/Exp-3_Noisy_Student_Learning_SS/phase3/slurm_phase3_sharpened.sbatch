#!/bin/bash
#SBATCH --job-name=p3-sharp-labels
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Phase 3: Pseudo-label sharpening sweep (4 configs)
#   0: thr=0.05, image_topk=64  (sharper threshold)
#   1: thr=0.07, image_topk=64  (sharpest threshold)
#   2: thr=0.03, image_topk=30  (fewer classes retained)
#   3: thr=0.05, image_topk=30  (sharper + fewer classes)
#
# All use T=2.0, tile_filter_k=6 (Phase 2 optimal)
# Baseline was: thr=0.03, image_topk=64

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

TILE_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase1_tiles_lucas"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
BASE_OUT="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase3_sharpened"

# Config arrays
THRESHOLDS=(0.05 0.07 0.03 0.05)
IMAGE_TOPKS=(64 64 30 30)
RUN_NAMES=(thr005_topk64 thr007_topk64 thr003_topk30 thr005_topk30)

IDX=${SLURM_ARRAY_TASK_ID}
THR=${THRESHOLDS[$IDX]}
TOPK=${IMAGE_TOPKS[$IDX]}
RUN_NAME=${RUN_NAMES[$IDX]}

OUT_PATH="${BASE_OUT}/${RUN_NAME}/lucas_pseudo_labels.parquet"

echo "Config: ${RUN_NAME} (thr=${THR}, image_topk=${TOPK})"
echo "Output: ${OUT_PATH}"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase3/generate_pseudo_labels.py \
  --tile-parquet-root "${TILE_ROOT}" \
  --out-path "${OUT_PATH}" \
  --geo-mask-npy "${GEO_MASK}" \
  --temperature 2.0 \
  --threshold "${THR}" \
  --max-labels 10 \
  --tile-filter-k 6 \
  --image-topk "${TOPK}"

echo "Done: Phase 3 sharpened pseudo-labels (${RUN_NAME})"
