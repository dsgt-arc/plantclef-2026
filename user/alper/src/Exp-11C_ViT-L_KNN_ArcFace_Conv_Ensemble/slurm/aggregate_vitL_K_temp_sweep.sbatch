#!/bin/bash
# Focused temperature sweep at the locked-in best ViT-L config:
#   scheme K (4+5+6+7), thr=0.080, alpha=0.15
# Tests T ∈ {1.0, 1.25, 1.5, 1.75}.
#
# T=1.50 is the existing baseline (0.41034). Including it is a sanity check;
# the local re-run should produce the same number. T=1.0 / T=1.25 explore
# whether sharper softmax helps; T=1.75 explores slight flattening
# (T=2.0 was already tested and degenerated for scheme E).
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/aggregate_vitL_K_temp_sweep.sbatch

#SBATCH --job-name=vitL-K-temp
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --qos=embers
#SBATCH --partition=cpu-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_K_temp_sweep_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

# K only needs scales 4-7
for s in 4 5 6 7; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L scheme K temperature sweep (thr=0.080, alpha=0.15) ==="
echo "Output: ${OUT_DIR}"
echo "Reference: T=1.50 produced F1=0.41034 in the alpha sweep."

# multiscale_aggregate.py runs one temperature at a time
for T in 1.0 1.25 1.5 1.75; do
    echo ""
    echo "--- T=${T} ---"
    python "${SRC_ROOT}/multiscale_aggregate.py" \
        --scale4-dir "${PARQUET_ROOT}/scale_4x4.parquet" \
        --scale5-dir "${PARQUET_ROOT}/scale_5x5.parquet" \
        --scale6-dir "${PARQUET_ROOT}/scale_6x6.parquet" \
        --scale7-dir "${PARQUET_ROOT}/scale_7x7.parquet" \
        --model-path "${MODEL_PATH}" \
        --species-map "${SPECIES_MAP}" \
        --class-mask-npy "${GEO_MASK}" \
        --out-dir "${OUT_DIR}" \
        --temperature "${T}" \
        --alpha 0.15 \
        --topk 10 \
        --image-topk 64 \
        --schemes "K" \
        --thresholds "0.080" \
        --csv-prefix "vitL_"
done

echo "=== Done. 4 CSVs in ${OUT_DIR}. ==="
echo "Anchor: T=1.50 should reproduce 0.41034 (sanity check)."
echo "Looking for T<1.5 or T=1.75 result that beats 0.41034."
