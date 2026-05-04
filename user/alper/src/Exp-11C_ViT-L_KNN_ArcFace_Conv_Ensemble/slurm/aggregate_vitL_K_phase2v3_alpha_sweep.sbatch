#!/bin/bash
# Phase2v3 standalone optimization: alpha sweep + tight threshold sweep at K, T=1.5.
#
# Locked so far: K (4+5+6+7), T=1.5
#   thr=0.075: 0.40013
#   thr=0.090: 0.40873  ← new best
#   thr=0.100: 0.40166
#   thr=0.115: 0.39523
# Sweeps alpha and re-tightens threshold around 0.090.
#
# Usage:
#   sbatch slurm/aggregate_vitL_K_phase2v3_alpha_sweep.sbatch

#SBATCH --job-name=vitL-K-alpha-p2v3
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

MODEL="dinov2_large"
OUTPUT_TAG="phase2v3"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_K_alpha_sweep_${MODEL}_${OUTPUT_TAG}"

MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

for s in 4 5 6 7; do
    f="${PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing scale ${s}x${s} parquet at $f"
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== Phase2v3 ViT-L K — alpha sweep + tight threshold (T=1.5) ==="
echo "Reference: K, T=1.5, alpha=0.15, thr=0.090 = 0.40873"
echo "Output: ${OUT_DIR}"

# 6 alphas × 5 thresholds = 30 CSVs
for ALPHA in 0.05 0.10 0.15 0.20 0.30 0.50; do
    echo ""
    echo "--- alpha=${ALPHA} ---"
    python "${SRC_ROOT}/multiscale_aggregate.py" \
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
        --schemes "K" \
        --thresholds "0.080,0.085,0.090,0.095,0.100" \
        --csv-prefix "vitL_p2v3_a${ALPHA}_"
done

echo ""
echo "=== Done. 30 CSVs in ${OUT_DIR} ==="
echo "Submission strategy:"
echo "  Round 1 (6 LB): one CSV per alpha at thr=0.090 → identify alpha*"
echo "  Round 2 (4 LB): at alpha*, sweep remaining 4 thresholds"
echo "  Round 3 (optional): if peak is at edge, extend further"
