#!/bin/bash
# Follow-up to aggregate_vitL_alpha_sweep.sbatch:
# tests J and L (the 3x3-containing schemes) at higher thresholds 0.100 and 0.115
# across all 5 alphas. The hypothesis is that 3x3 (coarser scale) shifts the
# optimal threshold upward; ViT-B saw E thr=0.100 -> L thr=0.125 (+0.025).
# Original alpha sweep covered only thr=0.080.
#
# Schemes:
#   J (3+4+5+6, 86 tiles)
#   L (3+4+5+6+7, 135 tiles)
# Alphas:     {0.05, 0.10, 0.15, 0.20, 0.30}
# Thresholds: {0.100, 0.115}
# Total: 2 schemes * 2 thresholds * 5 alphas = 20 CSVs
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/aggregate_vitL_JL_high_thr.sbatch

#SBATCH --job-name=vitL-JL-thr
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
# Reuse the existing alpha-sweep submissions dir so all alpha-sweep CSVs land together
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_alpha_sweep_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

for s in 3 4 5 6 7; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L follow-up: schemes J,L at thr in {0.100, 0.115} (T=1.5) ==="
echo "Output: ${OUT_DIR}"

for ALPHA in 0.05 0.10 0.15 0.20 0.30; do
    echo ""
    echo "--- alpha=${ALPHA} | J,L @ thr in {0.100, 0.115} ---"
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
        --alpha "${ALPHA}" \
        --topk 10 \
        --image-topk 64 \
        --schemes "J,L" \
        --thresholds "0.100,0.115" \
        --csv-prefix "vitL_"
done

echo "=== Done. 20 CSVs in ${OUT_DIR} (5 alphas × 2 schemes × 2 thresholds). ==="
echo "Reference: vitL scheme K thr=0.080 alpha=0.15 = 0.41034 (current best)."
echo "Looking for: vitL scheme L thr=0.100 or 0.115 with any alpha that beats 0.41034."