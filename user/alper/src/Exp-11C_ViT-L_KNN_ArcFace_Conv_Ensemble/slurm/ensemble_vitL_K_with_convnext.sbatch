#!/bin/bash
# Ensemble: ViT-L scheme K + ConvNeXt-L (scheme E or K).
# Uses ensemble_with_knn.py with the ViT-L K fused parquet substituted in
# for the "vit-knn-parquet" slot (the script blends two pre-fused vectors;
# whether the ViT branch came from a kNN pipeline or not is irrelevant).
#
# Tests both ConvNeXt scheme options:
#   E (4+5+6, 77 tiles) — Exp-11B production scheme; smaller, may diverge from ViT-L K
#   K (4+5+6+7, 126 tiles) — matched to ViT-L; should align better with ViT-L's predictions
#
# Weight sweep w ∈ {0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85}
# (w is ViT-L weight; 1-w is ConvNeXt weight)
# Reference: Exp-11B used w=0.77 with ViT-B+kNN branch (F1=0.44428).
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/ensemble_vitL_K_with_convnext.sbatch

#SBATCH --job-name=ens-vitLK
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --qos=embers
#SBATCH --partition=cpu-medium
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

# ViT-L K fused parquet (already produced by aggregate_vitL_alpha_sweep.sbatch at alpha=0.15)
VITL_FUSED_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_alpha_sweep_${MODEL}_${OUTPUT_TAG}/vitL_scheme_K_T1.50_fused_a0.15.parquet"

# ConvNeXt parquet locations
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR_BASE="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_ensemble_vitL_K"

# ---- Validate inputs ----
if [ ! -f "${VITL_FUSED_PARQUET}" ]; then
    echo "ERROR: missing ViT-L K fused parquet at ${VITL_FUSED_PARQUET}"
    echo "Run aggregate_vitL_alpha_sweep.sbatch first (it produces this as a side effect)."
    exit 1
fi
for s in 4 5 6 7; do
    f="${CNXT_PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing ConvNeXt scale ${s}x${s} parquet at $f"
        exit 1
    fi
done

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR_BASE}"

# ----------------------------------------------------------------------
# Branch 1: ConvNeXt scheme E (4+5+6) — Exp-11B production scheme
# ----------------------------------------------------------------------
echo ""
echo "=== ENSEMBLE 1/2: ViT-L K  +  ConvNeXt E (4+5+6) ==="
OUT_DIR_E="${OUT_DIR_BASE}_cnxtE"
mkdir -p "${OUT_DIR_E}"

python "${SRC_ROOT}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VITL_FUSED_PARQUET}" \
    --convnext-scales \
        "${CNXT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR_E}" \
    --temperature 1.5 \
    --cnxt-alpha 0.30 \
    --weights "0.65,0.70,0.75,0.78,0.80,0.82,0.85" \
    --thresholds "0.075,0.080,0.100" \
    --topk 10

# Tag CSVs so they don't collide with branch 2 when downloaded
for f in "${OUT_DIR_E}"/ens_knn_*.csv; do
    [ -e "$f" ] || continue
    new_name="$(basename "$f" .csv)_vitLK_cnxtE.csv"
    mv "$f" "${OUT_DIR_E}/${new_name}"
done

# ----------------------------------------------------------------------
# Branch 2: ConvNeXt scheme K (4+5+6+7) — matched to ViT-L K
# ----------------------------------------------------------------------
echo ""
echo "=== ENSEMBLE 2/2: ViT-L K  +  ConvNeXt K (4+5+6+7) ==="
OUT_DIR_K="${OUT_DIR_BASE}_cnxtK"
mkdir -p "${OUT_DIR_K}"

python "${SRC_ROOT}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VITL_FUSED_PARQUET}" \
    --convnext-scales \
        "${CNXT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR_K}" \
    --temperature 1.5 \
    --cnxt-alpha 0.30 \
    --weights "0.65,0.70,0.75,0.78,0.80,0.82,0.85" \
    --thresholds "0.075,0.080,0.100" \
    --topk 10

for f in "${OUT_DIR_K}"/ens_knn_*.csv; do
    [ -e "$f" ] || continue
    new_name="$(basename "$f" .csv)_vitLK_cnxtK.csv"
    mv "$f" "${OUT_DIR_K}/${new_name}"
done

echo ""
echo "=== Done. ==="
echo "Branch E (ViT-L K + ConvNeXt E): 7 weights × 3 thresholds = 21 CSVs in ${OUT_DIR_E}"
echo "Branch K (ViT-L K + ConvNeXt K): 7 weights × 3 thresholds = 21 CSVs in ${OUT_DIR_K}"
echo "Total: 42 CSVs."
echo ""
echo "Reference points (Exp-11 Phase 2, no-kNN baselines — apples-to-apples):"
echo "  ViT-B (E) + ConvNeXt (E), w=0.78 = 0.4224  <- direct baseline"
echo "  ViT-B (E) + ConvNeXt (3+4+5+6+7), w=0.78 = 0.4167  (extra scales hurt for ViT-B)"
echo ""
echo "Reference points (with-kNN, NOT directly comparable to this run):"
echo "  ViT-B + kNN + ConvNeXt (E), w=0.78 = 0.4354  (Exp-11 Approach A)"
echo "  ViT-B + LoRA-ArcFace-kNN + ConvNeXt (E), w=0.77 = 0.44428  (Exp-11B production)"
echo ""
echo "Predicted ViT-L K + ConvNeXt (E) ~ 0.428-0.430 (+0.006 over ViT-B baseline)"
echo "Predicted ViT-L K + ConvNeXt (K) potentially higher if matched-scheme rule holds"
