#!/bin/bash
# Phase2v3 standalone optimization: temperature sweep at scheme K.
#
# Standalone results (T=1.5, alpha=0.15) found K thr=0.100 = 0.40166 as best.
# This sweep explores T (with thr already narrowed to {0.090, 0.100, 0.115})
# at fixed alpha=0.15.
#
# Mixup/CutMix-trained models often want higher T (smoother softmax). Worth
# testing up to T=2.5 since phase2v3 cooldown saw mixup throughout.
#
# Usage:
#   sbatch slurm/aggregate_vitL_K_phase2v3_T_sweep.sbatch

#SBATCH --job-name=vitL-K-Tsweep-p2v3
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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_K_T_sweep_${MODEL}_${OUTPUT_TAG}"

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

echo "=== Phase2v3 ViT-L scheme K — temperature sweep ==="
echo "Reference: T=1.50, alpha=0.15, thr=0.100 = F1=0.40166"
echo "Output: ${OUT_DIR}"

# T sweep, each invocation produces 3 CSVs (thresholds 0.090/0.100/0.115)
for T in 1.00 1.25 1.50 1.75 2.00 2.50; do
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
        --thresholds "0.090,0.100,0.115" \
        --csv-prefix "vitL_p2v3_"
done

echo ""
echo "=== Done ==="
echo "Total: 6 temps × 3 thresholds = 18 CSVs"
echo "Submission: pick winning T at thr=0.100 first (6 LB), then expand around it."
