#!/bin/bash
# Exp-14C — isolate ViT-L + kNN performance (no ConvNeXt blend) before
# moving on to the full Stage 1ext sweep.
#
# Calls ensemble_v2.py with --weights "1.0" so ConvNeXt gets zero weight
# in the blend. Output is ViT_branch_fused = (kNN-blended classifier softmax,
# aggregated per image, with temporal fusion from controller default).
#
# Sweeps all 7 prefusion betas × 5 thresholds = 35 CSVs.
# Submit ~7 to LB (one per beta at thr=0.085) to characterize ViT+kNN curve.
#
# Usage:
#   sbatch slurm/knn_only_sweep.sbatch

#SBATCH --job-name=exp14c-knn-only
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP14_SCRIPTS="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion/scripts"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

cd "${EXP14_ROOT}"

PREFUSION_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_BASE="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_knn_only"

mkdir -p "${OUT_BASE}"

# Sanity checks
for b in 0.55 0.60 0.65 0.70 0.75 0.85 0.95; do
    f="${PREFUSION_DIR}/vitL_prefusion_b${b}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing prefusion parquet at $f"
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: ViT-L + kNN ONLY (w=1.0, no ConvNeXt) ==="
echo "Sweeping 7 betas × 5 thresholds = 35 CSVs"

for BETA in 0.55 0.60 0.65 0.70 0.75 0.85 0.95; do
    PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b${BETA}.parquet"
    OUT_DIR="${OUT_BASE}/b${BETA}"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "------------------------------------------------------------"
    echo "  beta_multi=${BETA}  →  ${OUT_DIR}"
    echo "------------------------------------------------------------"

    python "${EXP14_SCRIPTS}/ensemble_v2.py" \
        --vit-prefusion-parquet "${PREFUSION}" \
        --convnext-scales \
            "${CNXT_PARQUET_ROOT}/scale_4x4.parquet" \
            "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
            "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
            "${CNXT_PARQUET_ROOT}/scale_7x7.parquet" \
        --species-map "${SPECIES_MAP}" \
        --class-mask-npy "${GEO_MASK}" \
        --out-dir "${OUT_DIR}" \
        --temperature 1.5 \
        --weights "1.0" \
        --thresholds "0.075,0.085,0.095,0.100,0.115" \
        --topk 10 \
        --vit-aggregator max \
        --cnxt-aggregator max \
        --tag "exp14c_knnonly_b${BETA}"

    # Tag CSVs to distinguish from full-ensemble outputs
    for f in "${OUT_DIR}"/exp14c_knnonly_b${BETA}_*.csv; do
        [ -e "$f" ] || continue
        base="$(basename "$f" .csv)"
        case "$base" in
            *_vitLK_only) ;;
            *) mv "$f" "${OUT_DIR}/${base}_vitLK_only.csv" ;;
        esac
    done

    rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true
done

echo ""
echo "=== Done. 35 CSVs under ${OUT_BASE}/b{0.55,0.60,0.65,0.70,0.75,0.85,0.95}/ ==="
echo ""
echo "Submission strategy (~7 LB):"
echo "  Round 1: one CSV per beta at thr=0.085 (matches standalone-locked threshold)"
echo "  → identifies winning beta for ViT+kNN"
echo "  → tells us the kNN contribution before ConvNeXt adds noise"
echo ""
echo "Compare to standalone ViT-L (no kNN): 0.41186"
echo "Reference (Exp-9B ViT-B+kNN, beta=0.85): 0.42267"
echo "If ViT-L+kNN >> 0.42267: kNN is helping more here than in Exp-9B"
echo "If ViT-L+kNN ~ standalone: kNN is barely helping (problem to diagnose)"
