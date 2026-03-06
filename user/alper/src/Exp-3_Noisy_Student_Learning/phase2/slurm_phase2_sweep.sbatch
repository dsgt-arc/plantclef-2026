#!/bin/bash
#SBATCH --job-name=p2-temp-sweep
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/phase2_sweep"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase2/temperature_sweep.py \
  --tile-parquet-root "${PARQUET_ROOT}/phase1_tiles_test" \
  --out-dir "${OUT_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --temperatures 1.0,1.5,2.0,3.0 \
  --aug-filter \
  --tile-filter-k 16 \
  --topk 10

echo "Done: temperature sweep"
