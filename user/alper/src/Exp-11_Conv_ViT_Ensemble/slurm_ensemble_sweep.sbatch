#!/bin/bash
# Exp-11: Aggregate ConvNeXt + ViT tiles and sweep ensemble weights
# CPU-only job (no GPU needed for aggregation)

#SBATCH --job-name=ensemble
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ensemble_%A.out
#SBATCH --error=logs/ensemble_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

# Tile directories
CNXT_TILES="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
VIT_TILES="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale/scale_4x4.parquet"

# Mapping and masking
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

# Output
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_ensemble"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

echo "=== Exp-11: ConvNeXt + ViT Ensemble Sweep ==="
echo "ConvNeXt tiles: ${CNXT_TILES}"
echo "ViT tiles: ${VIT_TILES}"
echo "Output: ${OUT_DIR}"
echo "Date: $(date)"

python "${EXP11_DIR}/ensemble_sweep.py" \
    --convnext-tiles "${CNXT_TILES}" \
    --vit-tiles "${VIT_TILES}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --weights "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --thresholds "0.04,0.05,0.075,0.100,0.125" \
    --topk 10

echo "=== Ensemble sweep complete ==="
echo "Submissions in: ${OUT_DIR}"
