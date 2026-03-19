#!/bin/bash
#SBATCH --job-name=exp4-jpeg85-pipeline
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
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

# Input: JPEG85-compressed test embeddings (from slurm_extract_test_jpeg85.sbatch)
PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
AGG_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step9_jpeg85_agg"
FUSE_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step10_jpeg85_temporal_fusion"

# ─── Step A: View ablation sweep on JPEG85 embeddings ───
# Runs 1v, 2v_hflip, 2v_rot90, 3v x k={3,4,5,6} x T=1.5
# We mainly care about 1v results, but extra configs are free validation data
echo "=== Step A: View ablation on JPEG85 embeddings ==="
echo "Parquet root: ${PARQUET_ROOT}"
echo "Output: ${AGG_DIR}"
echo ""

python "${SRC_ROOT}/view_ablation_sweep.py" \
  --tile-parquet-root "${PARQUET_ROOT}" \
  --out-dir "${AGG_DIR}" \
  --model-path "${MODEL_PATH}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --agg-method "max_prob" \
  --temperatures "1.5" \
  --tile-filter-ks "3,4,5,6" \
  --thresholds "0.03" \
  --topk 10

# ─── Step B: Temporal fusion on 1v JPEG85 results ───
# Run tier6 (alpha 0.1-0.5) + tier7 (alpha 0.6-0.9) on k=3 and k=4
echo ""
echo "=== Step B: Temporal fusion on JPEG85 1v results ==="

for K in 3 4; do
  for TIER in tier6 tier7; do
    IN_PARQUET="${AGG_DIR}/mv_1v_max_prob_T1.5_k${K}.parquet"
    OUT_FUSE="${FUSE_DIR}/k${K}"

    echo ""
    echo "--- Temporal fusion: 1v k=${K} ${TIER} ---"
    echo "Input: ${IN_PARQUET}"
    echo "Output: ${OUT_FUSE}"

    python "${SRC_ROOT}/temporal_fusion.py" \
      --in-parquet "${IN_PARQUET}" \
      --out-dir "${OUT_FUSE}" \
      --species-map "${SPECIES_MAP}" \
      --class-mask-npy "${GEO_MASK}" \
      --topk 10 \
      --threshold 0.03 \
      --tier "${TIER}"
  done
done

echo ""
echo "=== JPEG85 pipeline complete ==="
echo "Aggregation results: ${AGG_DIR}/"
echo "Temporal fusion results: ${FUSE_DIR}/"
echo ""
echo "Key CSVs (alpha 0.1-0.9 for both k=3 and k=4):"
echo "  ${FUSE_DIR}/k3/tf_A_blend_a*_1v_thr0.03_top10.csv"
echo "  ${FUSE_DIR}/k4/tf_A_blend_a*_1v_thr0.03_top10.csv"
echo ""
echo "JPEG85 baselines (no fusion):"
echo "  1v k3: ${AGG_DIR}/mv_1v_max_prob_T1.5_k3_thr0.03_top10.csv (F1 0.33073)"
echo "  1v k4: ${AGG_DIR}/mv_1v_max_prob_T1.5_k4_thr0.03_top10.csv (F1 0.32717)"
echo ""
echo "Non-JPEG baselines:"
echo "  1v k4 + alpha=0.1: F1 0.33946 (previous best)"
echo "  JPEG85 1v k3 + alpha=0.5: F1 0.35486 (current best)"
