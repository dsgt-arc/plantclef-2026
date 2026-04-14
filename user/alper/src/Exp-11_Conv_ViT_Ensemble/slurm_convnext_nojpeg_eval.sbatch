#!/bin/bash
# Evaluate ConvNeXt standalone without JPEG — compare against JPEG85 version

#SBATCH --job-name=cnxt-nojpg-eval
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cnxt_nojpeg_eval_%A.out
#SBATCH --error=logs/cnxt_nojpeg_eval_%A.err

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

CNXT_TILES="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large_nojpeg"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_nojpeg"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

# Dummy vit-agg not needed — we only care about convnext_only
# Use the existing vit agg just to satisfy the script, weights=0.0 means convnext only
python "${EXP11_DIR}/ensemble_sweep.py" \
    --convnext-tiles "${CNXT_TILES}" \
    --vit-tiles "${CNXT_TILES}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --weights "0.0" \
    --thresholds "0.03,0.04,0.05,0.075,0.100" \
    --topk 10

echo "=== Done. Compare against JPEG85 ConvNeXt results ==="
