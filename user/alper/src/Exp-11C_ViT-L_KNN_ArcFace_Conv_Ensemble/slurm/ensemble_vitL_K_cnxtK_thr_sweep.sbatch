#!/bin/bash
# Fine-grained threshold sweep at the winning ensemble config:
#   ViT-L K (4+5+6+7, alpha=0.15) + ConvNeXt K (4+5+6+7, alpha=0.30)
# Weight is essentially flat in [0.70, 0.75]; threshold is the dominant lever.
# Existing data: thr=0.080 -> 0.41461,  thr=0.100 -> 0.42540 (+0.011 from threshold).
# Optimum is almost certainly above thr=0.100 — this sweep finds where the curve peaks.
#
# Sweep:
#   Weights:    w in {0.70, 0.75}  (both inside the noise-flat band)
#   Thresholds: thr in {0.085, 0.090, 0.095, 0.105, 0.110, 0.115, 0.120,
#                       0.125, 0.130, 0.140, 0.150}
# Total: 2 × 11 = 22 CSVs.
#
# Usage:
#   sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/ensemble_vitL_K_cnxtK_thr_sweep.sbatch

#SBATCH --job-name=ens-vitLK-thr
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

VITL_FUSED_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_vitL_alpha_sweep_${MODEL}_${OUTPUT_TAG}/vitL_scheme_K_T1.50_fused_a0.15.parquet"
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11c_ensemble_vitL_K_cnxtK_thr_sweep"

if [ ! -f "${VITL_FUSED_PARQUET}" ]; then
    echo "ERROR: missing ViT-L K fused parquet at ${VITL_FUSED_PARQUET}"
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
mkdir -p "${SRC_ROOT}/logs" "${OUT_DIR}"

echo "=== ViT-L K + ConvNeXt K — fine threshold sweep ==="
echo "Existing best: w=0.75 thr=0.100 = 0.42540"
echo "Output: ${OUT_DIR}"

python "${SRC_ROOT}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VITL_FUSED_PARQUET}" \
    --convnext-scales \
        "${CNXT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --cnxt-alpha 0.30 \
    --weights "0.70,0.75" \
    --thresholds "0.085,0.090,0.095,0.105,0.110,0.115,0.120,0.125,0.130,0.140,0.150" \
    --topk 10

# Tag CSVs
for f in "${OUT_DIR}"/ens_knn_*.csv; do
    [ -e "$f" ] || continue
    base="$(basename "$f" .csv)"
    case "$base" in
        *_vitLK_cnxtK) ;;
        *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtK.csv" ;;
    esac
done

echo "=== Done. ==="
echo "Total: 22 CSVs (2 weights × 11 thresholds)."
echo ""
echo "Submission strategy: scan thr=0.105, 0.110, 0.115, 0.120, 0.125 first (most likely peak)."
echo "Reference points:"
echo "  This run target: beat 0.42540 (w=0.75 thr=0.100)"
echo "  Stretch goal: approach 0.4354 (Exp-11 with-kNN baseline)"
