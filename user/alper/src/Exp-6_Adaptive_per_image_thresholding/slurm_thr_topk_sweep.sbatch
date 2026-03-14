#!/bin/bash
#SBATCH --job-name=exp6-thr-topk
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-6_Adaptive_per_image_thresholding"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

# Best Exp-4 temporal fusion parquet
IN_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step10_jpeg85_temporal_fusion/k3/tf_A_blend_a0.6_1v.parquet"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp6_adaptive_threshold/thr_topk_sweep"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

echo "=== Exp-6: Threshold x TopK Joint Sweep ==="

python "${SRC_ROOT}/threshold_topk_sweep.py" \
  --in-parquet "${IN_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --topks "3,4,5,6,7" \
  --thresholds "0.02,0.025,0.03,0.035,0.04,0.045,0.05"

echo ""
echo "=== Sweep complete ==="
echo "35 CSVs in: ${OUT_DIR}/"
