#!/bin/bash
# Single-scale 4x4 gate test: phase2v3 vs phase2v2 on the LB.
# Aggregates one parquet (scale 4x4) with scheme A and the standard
# standalone params. Submits one CSV per threshold; pick thr=0.080
# for direct comparison to phase2v2.
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v3 slurm/aggregate_vitL_4x4_gate.sbatch
#
# Prerequisite: scale_4x4.parquet must exist under
#   /storage/.../aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}/

#SBATCH --job-name=vitL-4x4-gate
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

MODEL="${MODEL:-dinov2_large}"
OUTPUT_TAG="${OUTPUT_TAG:-phase2v3}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp13_vit_${MODEL}_${OUTPUT_TAG}"
SCALE4_PARQUET="${PARQUET_ROOT}/scale_4x4.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_4x4_gate_${MODEL}_${OUTPUT_TAG}"

# Pretrained ViT-B path used only for species-map dim check inside multiscale_aggregate.py
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

if [ ! -f "${SCALE4_PARQUET}" ]; then
    echo "ERROR: missing 4x4 parquet at ${SCALE4_PARQUET}"
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L 4x4-only gate (scheme A) ==="
echo "Model:    ${MODEL} / ${OUTPUT_TAG}"
echo "Parquet:  ${SCALE4_PARQUET}"
echo "Output:   ${OUT_DIR}"

python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale4-dir "${SCALE4_PARQUET}" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --topk 10 \
    --image-topk 64 \
    --schemes "A" \
    --thresholds "0.075,0.080,0.100" \
    --csv-prefix "vitL_4x4_${OUTPUT_TAG}_"

echo "=== Done. CSVs in ${OUT_DIR} ==="
echo "Submit the thr=0.080 CSV for direct comparison to phase2v2 4x4 baseline."
