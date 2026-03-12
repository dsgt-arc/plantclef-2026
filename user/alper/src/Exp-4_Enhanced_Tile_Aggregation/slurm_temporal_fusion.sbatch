#!/bin/bash
#SBATCH --job-name=exp4-temporal-fusion
#SBATCH --account=paceship-dsgt_clef2026
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

# Input: best Exp-4 aggregated parquet (per_view_or + max_prob + T=1.5 + k=3)
IN_PARQUET="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/aerten3/submissions/exp4_multiview/step5_topm_prob/mv_per_view_or_top1_prob_T1.5_k3.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp4_multiview/step7_temporal_fusion"

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
  --tier tier4

echo ""
echo "=== Temporal fusion sweep complete ==="
echo "Submissions in: ${OUT_DIR}/"
echo "Tier 4: blend alpha={0.2,0.3,0.4,0.45} + blend+boost at alpha={0.4,0.5}"
echo "Best so far: A_blend alpha=0.5 = F1 0.33403 (baseline 0.31487)"
echo "Submit CSVs to Kaggle to find optimal alpha"
