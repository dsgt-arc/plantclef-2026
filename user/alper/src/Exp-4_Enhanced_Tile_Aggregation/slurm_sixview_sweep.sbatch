#!/bin/bash
#SBATCH --job-name=exp4-sixview
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

SRC_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-4_Enhanced_Tile_Aggregation"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test"
ADDITIONAL_VIEWS="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_additional_views/additional_views.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step6_sixview"

# Step 6: Six-view aggregation sweep
# Compare 3-view (baseline 0.315) vs 6-view per_view_or with max_prob
# {3view, 6view} x max_prob x T=1.5 x k={3,4,5,6}
echo "=== Step 6: Six-view aggregation sweep ==="
echo "Original tiles: ${PARQUET_ROOT}"
echo "Additional views: ${ADDITIONAL_VIEWS}"
echo "Output: ${OUT_DIR}"

python "${SRC_ROOT}/sixview_sweep.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
  --additional-views-parquet "${ADDITIONAL_VIEWS}" \
  --out-dir "${OUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --agg-method "max_prob" \
  --temperatures "1.5" \
  --tile-filter-ks "3,4,5,6" \
  --thresholds "0.03" \
  --topk 10

echo ""
echo "=== Six-view sweep complete ==="
echo "Submissions in: ${OUT_DIR}/"
echo "Compare 3view vs 6view results on Kaggle"
echo "Baseline: 3-view per_view_or + max_prob T1.5 k3 = F1 0.31487"
