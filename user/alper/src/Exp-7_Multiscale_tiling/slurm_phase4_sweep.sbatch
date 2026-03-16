#!/bin/bash
#SBATCH --job-name=exp7-phase4-sweep
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Exp-7 Phase 4: Alpha + Temperature re-tuning on Scheme E (4x4+5x5+6x6)
# Current best: alpha=0.6, T=1.5, thr=0.075, F1=0.38466
#
# Step 1: Sweep alpha (0.4, 0.5, 0.6, 0.7, 0.8) with T=1.5 fixed
# Step 2: Sweep temperature (1.0, 1.25, 1.5, 1.75, 2.0) with alpha=0.6 fixed
#
# CSV naming: scheme_E_T{temp}_a{alpha}_thr0.075_top10.csv

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-7_Multiscale_tiling"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

echo "Running on host: $(hostname)"
echo "Job: ${SLURM_JOB_NAME}  JobID: ${SLURM_JOB_ID}"

# Paths
SCALE4_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
SCALE5_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
SCALE6_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp7_multiscale/phase4"

echo "=== Exp-7 Phase 4: Alpha + Temperature Sweep ==="
echo "Output: ${OUT_DIR}"

# ── Step 1: Alpha sweep (T=1.5 fixed) ──
echo ""
echo "=== Step 1: Alpha sweep (T=1.5, thr=0.075) ==="

for ALPHA in 0.4 0.5 0.6 0.7 0.8; do
  echo "--- Alpha=${ALPHA} ---"
  python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale4-dir "${SCALE4_DIR}" \
    --scale5-dir "${SCALE5_DIR}" \
    --scale6-dir "${SCALE6_DIR}" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha "${ALPHA}" \
    --topk 10 \
    --image-topk 64 \
    --tile-filter-k 0 \
    --schemes E \
    --thresholds "0.075"
done

# ── Step 2: Temperature sweep (alpha=0.6 fixed) ──
echo ""
echo "=== Step 2: Temperature sweep (alpha=0.6, thr=0.075) ==="

for TEMP in 1.0 1.25 1.5 1.75 2.0; do
  echo "--- T=${TEMP} ---"
  python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale4-dir "${SCALE4_DIR}" \
    --scale5-dir "${SCALE5_DIR}" \
    --scale6-dir "${SCALE6_DIR}" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature "${TEMP}" \
    --alpha 0.6 \
    --topk 10 \
    --image-topk 64 \
    --tile-filter-k 0 \
    --schemes E \
    --thresholds "0.075"
done

echo ""
echo "=== Phase 4 sweep complete ==="
echo "All CSVs in: ${OUT_DIR}"
echo "Baseline: scheme_E_T1.50_a0.6_thr0.075_top10.csv = F1 0.38466"
