#!/bin/bash
# Aggregate ViT-L multiscale tiles → submission CSVs for Kaggle
# (standalone ViT-L performance, no ensemble/kNN yet).
#
# Tests two tile schemes:
#   Scheme E: 4+5+6 = 77 tiles  (matches ViT-B production, apples-to-apples vs ViT-B 0.42267)
#   Scheme K: 3+4+5+6+7 = 135 tiles  (richer multiscale, mirrors ConvNext sweep in Exp-11)
#
# Prerequisite: 5 scale parquets must exist under
#   /storage/.../aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}/scale_NxN.parquet
# Run extract_multiscale_3to7.sbatch first.
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/aggregate_vitL_standalone.sbatch

#SBATCH --job-name=vitL-agg-standalone
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

MODEL="${MODEL:-dinov2_large}"
OUTPUT_TAG="${OUTPUT_TAG:-phase2v2}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${SRC_ROOT}/submissions/vitL_standalone_${MODEL}_${OUTPUT_TAG}"

# Reuse pretrained ViT-B checkpoint just to load the species map header
# (multiscale_aggregate.py requires --model-path but only uses it for the species map dim check)
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

# ---- Validate scale parquets ----
for s in 3 4 5 6 7; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        echo "Run extract_multiscale_3to7.sbatch first."
        exit 1
    fi
done

# ---- Environment ----
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L standalone aggregation ==="
echo "Model:    ${MODEL} / ${OUTPUT_TAG}"
echo "Parquets: ${PARQUET_ROOT}"
echo "Output:   ${OUT_DIR}"

# Sweep schemes E (4+5+6 = 77 tiles) and K (3+4+5+6+7 = 135 tiles) at thresholds 0.075–0.150
python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale3-dir "${PARQUET_ROOT}/scale_3x3.parquet" \
    --scale4-dir "${PARQUET_ROOT}/scale_4x4.parquet" \
    --scale5-dir "${PARQUET_ROOT}/scale_5x5.parquet" \
    --scale6-dir "${PARQUET_ROOT}/scale_6x6.parquet" \
    --scale7-dir "${PARQUET_ROOT}/scale_7x7.parquet" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --topk 10 \
    --image-topk 64 \
    --schemes "A,E,J,K,L" \
    --thresholds "0.075,0.100,0.125,0.150"

# To compare fairly against ViT-B, run aggregate_vitB_standalone.sbatch
# (same alpha/thresholds/schemes) and submit the matching CSVs side-by-side.
echo "=== Done. Submit CSVs in ${OUT_DIR} to Kaggle. ==="
echo "Run aggregate_vitB_standalone.sbatch with same params for matched ViT-B comparison."
