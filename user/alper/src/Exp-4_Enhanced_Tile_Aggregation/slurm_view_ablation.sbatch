#!/bin/bash
#SBATCH --job-name=exp4-view-ablation
#SBATCH --account=paceship-clef2026_plantclef
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
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step7_view_ablation"

# Step 7: View ablation — does each augmentation view help?
# 1v (orig only), 2v_hflip (orig+hflip), 2v_rot90 (orig+rot90), 3v (all)
# All with max_prob + T=1.5 + k={3,4,5,6}
# 3v results already exist from Step 4 (included for cross-validation)
echo "=== Step 7: Incremental view ablation ==="
echo "Configs: 1v, 2v_hflip, 2v_rot90, 3v"
echo "Sweep: 4 view configs x 4 k values = 16 CSVs"

python "${SRC_ROOT}/view_ablation_sweep.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
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
echo "=== View ablation complete ==="
echo "Submissions in: ${OUT_DIR}/"
echo "Compare 1v vs 2v vs 3v to isolate TTA contribution"
echo "3v baseline: per_view_or + max_prob T1.5 k3 = F1 0.31487"
