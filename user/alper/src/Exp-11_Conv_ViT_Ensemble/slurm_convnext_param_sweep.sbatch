#!/bin/bash
# Exp-11: ConvNeXt standalone parameter sweep
# Sweep temperature, alpha, threshold to optimize ConvNeXt before ensembling

#SBATCH --job-name=cnxt-sweep
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cnxt_sweep_%A.out
#SBATCH --error=logs/cnxt_sweep_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"

CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_cnxt_param_sweep"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

echo "=== ConvNeXt Parameter Sweep ==="
echo "Date: $(date)"

python "${EXP11_DIR}/convnext_param_sweep.py" \
    --convnext-4x4 "${CNXT_DIR}/scale_4x4.parquet" \
    --convnext-5x5 "${CNXT_DIR}/scale_5x5.parquet" \
    --convnext-6x6 "${CNXT_DIR}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperatures "1.0,1.5,2.0,2.5" \
    --alphas "0.05,0.15,0.30" \
    --thresholds "0.04,0.075,0.100" \
    --topk 10

echo "=== Sweep complete ==="
