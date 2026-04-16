#!/bin/bash
#SBATCH --job-name=p1-lucas-agg
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

export PYSPARK_SUBMIT_ARGS="--driver-memory 48g pyspark-shell"

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k"
SHARD=$(printf "shard_%04d" ${SLURM_ARRAY_TASK_ID})
META="${PLANTCLEF_ROOT}/data/raw/metadata_lucas/LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

# Skip if already done
if [ -f "${PARQUET_ROOT}/phase1_teacher_agg_lucas/${SHARD}/_SUCCESS" ]; then
  echo "Already completed ${SHARD}"
  exit 0
fi

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase1/aggregate_teacher.py \
  --in-parquet-root "${PARQUET_ROOT}/phase1_tiles_lucas/${SHARD}" \
  --metadata-csv "${META}" \
  --out-parquet "${PARQUET_ROOT}/phase1_teacher_agg_lucas/${SHARD}" \
  --image-col image_name \
  --aug-filter \
  --keep-top-tiles 8 \
  --temperature 2.0 \
  --denoise-eps 1e-6 \
  --image-topk 64 \
  --geo-mask-npy "${GEO_MASK}"

find "${PARQUET_ROOT}/phase1_teacher_agg_lucas/${SHARD}" -name "*.crc" -type f -delete

echo "Done: ${SHARD}"
