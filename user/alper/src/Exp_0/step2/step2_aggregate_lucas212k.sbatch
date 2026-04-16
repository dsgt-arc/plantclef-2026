#!/bin/bash
#SBATCH --job-name=lucas212k-step2
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-212%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

ROOT=/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026
INROOT=${ROOT}/parquet/lucas_212k/feature_bank
OUTROOT=${ROOT}/parquet/lucas_212k/step2_teacher_agg
META=${ROOT}/data/raw/metadata_lucas_212k/training_metadata_212k.csv

SHARD=$(printf "shard_%04d" ${SLURM_ARRAY_TASK_ID})

mkdir -p logs
mkdir -p ${OUTROOT}

# Skip if shard already finished successfully
if [ -f "${OUTROOT}/${SHARD}/_SUCCESS" ]; then
  echo "Already completed ${SHARD}"
  exit 0
fi

# Give Spark enough JVM heap
export PYSPARK_SUBMIT_ARGS="--driver-memory 48g pyspark-shell"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step2/run_step2_teacher_aggregation.py \
  --in-parquet-root ${INROOT}/${SHARD} \
  --metadata-csv ${META} \
  --out-parquet ${OUTROOT}/${SHARD} \
  --temperature 2.0 \
  --denoise-eps 1e-6 \
  --image-topk 64


# Clean crc files
find ${OUTROOT}/${SHARD} -name "*.crc" -type f -delete