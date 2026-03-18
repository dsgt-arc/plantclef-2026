#!/bin/bash
#SBATCH --job-name=p1-test-agg
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=[aerten3@gatech.edu](mailto:aerten3@gatech.edu)

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

# Activate environment

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

# Spark driver memory

export PYSPARK_SUBMIT_ARGS="--driver-memory 48g pyspark-shell"

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

# -----------------------------

# Paths

# -----------------------------

PARQUET_BASE="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test"
TILE_ROOT="${PARQUET_BASE}/phase1_tiles_test"
AGG_ROOT="${PARQUET_BASE}/phase1_teacher_agg_test"

GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

echo "Tile parquet root: ${TILE_ROOT}"
echo "Aggregation output: ${AGG_ROOT}"

# -----------------------------

# Run aggregation

# -----------------------------

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase1/aggregate_teacher.py \
  --in-parquet-root "${TILE_ROOT}" \
  --out-parquet "${AGG_ROOT}" \
  --image-col image_name \
  --aug-filter \
  --keep-top-tiles 8 \
  --temperature 2.0 \
  --denoise-eps 1e-6 \
  --image-topk 64 \
  --geo-mask-npy "${GEO_MASK}"

# Remove Spark CRC files

find "${AGG_ROOT}" -name "*.crc" -type f -delete

echo "Aggregation finished"
echo "Output written to: ${AGG_ROOT}"
