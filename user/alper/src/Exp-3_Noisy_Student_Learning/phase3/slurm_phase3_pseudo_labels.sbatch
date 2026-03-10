#!/bin/bash
#SBATCH --job-name=p3-pseudo-labels
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

TILE_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase1_tiles_lucas"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_PATH="${PLANTCLEF_ROOT}/aerten3/phase3_student_dataset/lucas_pseudo_labels.parquet"

# Best temperature from Phase 2 sweep — UPDATE after running Phase 2
TEMPERATURE=2.0
THRESHOLD=0.1

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase3/generate_pseudo_labels.py \
  --tile-parquet-root "${TILE_ROOT}" \
  --out-path "${OUT_PATH}" \
  --geo-mask-npy "${GEO_MASK}" \
  --temperature "${TEMPERATURE}" \
  --threshold "${THRESHOLD}" \
  --max-labels 10 \
  --tile-filter-k 16 \
  --image-topk 64

echo "Done: Phase 3 pseudo-label generation"
