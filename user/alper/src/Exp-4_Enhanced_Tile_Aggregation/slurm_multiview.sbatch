#!/bin/bash
#SBATCH --job-name=exp4-multiview
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview"

# ── Step 1: Multi-view strategies with standard Noisy-OR ──
# Test all 4 view strategies at best-known config (T=2.0, k=6)
echo "=== Step 1: View strategy sweep (noisy_or) ==="
python "${SRC_ROOT}/multiview_aggregate.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
  --out-dir "${OUT_DIR}/step1_views" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --strategies "avg_logits,all_tiles,per_view_or,orig_only" \
  --aggregation-methods "noisy_or" \
  --temperatures "2.0" \
  --tile-filter-ks "6" \
  --thresholds "0.03" \
  --topk 10

# all_tiles triples effective tile count — sweep k since optimal k may differ
echo "=== Step 1b: all_tiles k sweep ==="
python "${SRC_ROOT}/multiview_aggregate.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
  --out-dir "${OUT_DIR}/step1b_alltiles_ksweep" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --strategies "all_tiles" \
  --aggregation-methods "noisy_or" \
  --temperatures "2.0" \
  --tile-filter-ks "2,4,6,8,10,12,16" \
  --thresholds "0.03" \
  --topk 10

# Temperature sweep for most promising multi-view strategies
echo "=== Step 1c: Temperature sweep for avg_logits + per_view_or ==="
python "${SRC_ROOT}/multiview_aggregate.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
  --out-dir "${OUT_DIR}/step1c_temp_sweep" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --strategies "avg_logits,per_view_or" \
  --aggregation-methods "noisy_or" \
  --temperatures "1.5,2.0,2.5" \
  --tile-filter-ks "4,6,8" \
  --thresholds "0.03" \
  --topk 10

# ── Step 3: Alternative aggregation methods ──
echo "=== Step 3: Aggregation method sweep ==="
python "${SRC_ROOT}/multiview_aggregate.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
  --out-dir "${OUT_DIR}/step3_agg_methods" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --strategies "avg_logits,orig_only" \
  --aggregation-methods "noisy_or,weighted_noisy_or,mean_prob,max_prob" \
  --temperatures "2.0" \
  --tile-filter-ks "6,10,16" \
  --thresholds "0.03" \
  --topk 10

echo ""
echo "=== All multi-view sweeps complete ==="
echo "Submissions in: ${OUT_DIR}/"
echo "Next: submit CSVs to Kaggle and compare vs baseline F1=0.28093"
