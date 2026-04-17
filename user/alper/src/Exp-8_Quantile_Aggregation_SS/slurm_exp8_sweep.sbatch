#!/bin/bash
#SBATCH --job-name=exp8-quantile-sweep
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-8: Quantile aggregation sweep on Scheme E (4x4+5x5+6x6)
#
# Baseline: max_prob, T=1.5, alpha=0.2, thr=0.075, F1=0.39420
#
# Sweep: max_prob vs quantile @ q={0.75, 0.80, 0.85, 0.90, 0.95}
#        x thresholds {0.04, 0.06, 0.075, 0.08, 0.10}
#
# CPU-only job — no GPU needed. Uses existing tile embeddings.
# Expected runtime: ~10 minutes.

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-8_Quantile_Aggregation"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

# Paths (same as Exp-7)
SCALE4_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
SCALE5_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
SCALE6_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp8_quantile_agg"

echo "=== Exp-8: Quantile Aggregation Sweep ==="
echo "Output: ${OUT_DIR}"

python "${SRC_ROOT}/multiscale_aggregate.py" \
  --scale4-dir "${SCALE4_DIR}" \
  --scale5-dir "${SCALE5_DIR}" \
  --scale6-dir "${SCALE6_DIR}" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --out-dir "${OUT_DIR}" \
  --temperature 1.5 \
  --alpha 0.2 \
  --topk 10 \
  --image-topk 64 \
  --tile-filter-k 0 \
  --schemes E \
  --agg-methods "max_prob,quantile" \
  --agg-quantiles "0.75,0.80,0.85,0.90,0.95" \
  --thresholds "0.04,0.06,0.075,0.08,0.10"

echo ""
echo "=== Exp-8 sweep complete ==="
echo "All CSVs in: ${OUT_DIR}"
echo "Baseline to beat: max_prob, alpha=0.2, thr=0.075 = F1 0.39420"
