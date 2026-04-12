#!/bin/bash
#SBATCH --job-name=exp12-aggregate
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-12: Border Cropping — Aggregate + temporal fusion + submission CSVs.
# Runs 4x4-only aggregation for baseline (no crop) + 3 crop levels (3%, 5%, 7%).
# Sweeps thresholds: 0.04, 0.07, 0.10 at alpha=0.6 (Exp-7 4x4 params).
#
# Prerequisites: slurm_crop_extract.sbatch must have completed.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

EXP7_SRC="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${EXP7_SRC}:${PYTHONPATH:-}"

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

# Paths
BASELINE_4x4="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
CROPPED_PARQUET_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp12_border_crop"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp12_border_crop"
mkdir -p "${OUT_DIR}"

# Use Exp-7 4x4 baseline parameters for fair comparison
TEMPERATURE=1.5
ALPHA=0.6
TOPK=10
IMAGE_TOPK=64

# Thresholds to sweep (matching Exp-7 baseline range)
THRESHOLDS="0.04,0.07,0.10"

echo "=== Exp-12: Border Crop 4x4 Comparison ==="
echo "Temperature: ${TEMPERATURE}, Alpha: ${ALPHA}, TopK: ${TOPK}"
echo "Thresholds:  ${THRESHOLDS}"
echo ""

# --------------------------------------------------------------------------
# Run baseline (no crop) — uses existing Exp-4 4x4 parquets
# --------------------------------------------------------------------------
echo "=== [0/4] Baseline (no crop) ==="
python "${EXP7_SRC}/multiscale_aggregate.py" \
  --scale4-dir "${BASELINE_4x4}" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}/baseline" \
  --temperature ${TEMPERATURE} \
  --alpha ${ALPHA} \
  --topk ${TOPK} \
  --image-topk ${IMAGE_TOPK} \
  --tile-filter-k 0 \
  --schemes A \
  --thresholds "${THRESHOLDS}"

# --------------------------------------------------------------------------
# Run each crop percentage
# --------------------------------------------------------------------------
for CROP_PCT in 3 5 7; do
  PARQUET_FILE="${CROPPED_PARQUET_DIR}/scale_4x4_crop${CROP_PCT}pct.parquet"

  if [ ! -f "${PARQUET_FILE}" ]; then
    echo "WARNING: ${PARQUET_FILE} not found, skipping crop ${CROP_PCT}%"
    continue
  fi

  echo ""
  echo "=== [crop ${CROP_PCT}%] ==="
  echo "Parquet: ${PARQUET_FILE}"

  python "${EXP7_SRC}/multiscale_aggregate.py" \
    --scale4-dir "${PARQUET_FILE}" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}/crop_${CROP_PCT}pct" \
    --temperature ${TEMPERATURE} \
    --alpha ${ALPHA} \
    --topk ${TOPK} \
    --image-topk ${IMAGE_TOPK} \
    --tile-filter-k 0 \
    --schemes A \
    --thresholds "${THRESHOLDS}"
done

echo ""
echo "=== All aggregations complete ==="
echo "Submission CSVs in: ${OUT_DIR}/"
echo ""
echo "Generated CSVs (submit to Kaggle for F1 comparison):"
find "${OUT_DIR}" -name "*.csv" | sort
echo ""
echo "Download these CSVs and submit to Kaggle to compare F1 scores."
