#!/bin/bash
# Exp-11: Ensemble with different ConvNeXt scale schemes
# Test if simpler ConvNeXt = better ensemble

#SBATCH --job-name=ens-scales
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ens_scale_sweep_%A.out
#SBATCH --error=logs/ens_scale_sweep_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
VIT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VIT_4x4="${VIT_DIR}/scale_4x4.parquet"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
BASE_OUT="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_ens_scale_sweep"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"

mkdir -p "${BASE_OUT}"

WEIGHTS="0.75,0.78,0.80,0.85,0.90"
VIT_SCALES="${VIT_4x4} ${VIT_DIR}/scale_5x5.parquet ${VIT_DIR}/scale_6x6.parquet"

run_scheme() {
    local TAG="$1"
    shift
    local CNXT_FILES="$@"

    echo ""
    echo "=== CNxt scheme: ${TAG} ==="
    python "${EXP11_DIR}/ensemble_optimized.py" \
        --convnext-scales ${CNXT_FILES} \
        --vit-scales ${VIT_SCALES} \
        --species-map "${SPECIES_MAP}" --class-mask-npy "${CLASS_MASK}" \
        --out-dir "${BASE_OUT}" \
        --prefix "${TAG}" \
        --weights "${WEIGHTS}" --thresholds "0.100" \
        --topk 10
}

echo "=== Ensemble Scale Sweep ==="
echo "Date: $(date)"

run_scheme "cnxt_4" \
    "${CNXT_DIR}/scale_4x4.parquet"

run_scheme "cnxt_45" \
    "${CNXT_DIR}/scale_4x4.parquet" "${CNXT_DIR}/scale_5x5.parquet"

run_scheme "cnxt_56" \
    "${CNXT_DIR}/scale_5x5.parquet" "${CNXT_DIR}/scale_6x6.parquet"

run_scheme "cnxt_6" \
    "${CNXT_DIR}/scale_6x6.parquet"

run_scheme "cnxt_456" \
    "${CNXT_DIR}/scale_4x4.parquet" "${CNXT_DIR}/scale_5x5.parquet" "${CNXT_DIR}/scale_6x6.parquet"

echo ""
echo "=== All schemes done ==="
