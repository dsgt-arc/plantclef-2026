#!/bin/bash
#SBATCH --job-name=exp4-temporal-fusion
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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

# Input: 1-view (no TTA) best parquet (orig_only + max_prob + T=1.5 + k=4 = F1 0.31950)
IN_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step7_view_ablation/mv_1v_max_prob_T1.5_k4.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step8_temporal_fusion_1v"

echo "=== Step 7: Temporal Fusion Sweep ==="
echo "Input parquet: ${IN_PARQUET}"
echo "Output dir: ${OUT_DIR}"
echo ""

python "${SRC_ROOT}/temporal_fusion.py" \
  --in-parquet "${IN_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --class-mask-npy "${GEO_MASK}" \
  --topk 10 \
  --threshold 0.03 \
  --tier tier6

echo ""
echo "=== Temporal fusion sweep complete ==="
echo "Submissions in: ${OUT_DIR}/"
echo "Tier 6: 1v (no TTA) base + alpha={0.1,0.2,0.3,0.4,0.5}"
echo "1v base: orig_only + max_prob + T1.5 + k4 = F1 0.31950"
echo "Previous best: 3v + temporal fusion alpha=0.1 = F1 0.33881"
