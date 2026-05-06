#!/bin/bash
# Exp-10C — standalone ConvNeXt aggregation: cooldown vs production.
#
# GATE TEST before ensemble integration: does the phase2v3 cooldown actually
# improve standalone ConvNeXt-Large quality? If standalone F1 is flat or
# worse, the cooldown didn't help and we should not waste LB on full pipeline.
#
# Aggregates BOTH parquet sets at the schemes the ensemble uses:
#   - Scheme P (5+6+7, 110 tiles)  — matches production ensemble's ConvNeXt branch
#   - Scheme K (4+5+6+7, 126 tiles) — richer baseline for comparison
#   - Scheme I (6 alone, 36 tiles)  — cheapest single-scale reference
#
# Each scheme produces CSVs at thresholds 0.075, 0.085, 0.100, 0.125 for
# both production and cooldown ConvNeXt. Compare side-by-side.
#
# Output:
#   ${PLANTCLEF_ROOT}/aerten3/submissions/exp10c_standalone_compare/
#     production_<scheme>_T1.50_thr<thr>_top10.csv  (production ConvNeXt)
#     cooldown_<scheme>_T1.50_thr<thr>_top10.csv    (cooldown ConvNeXt)
#
# Usage:
#   sbatch slurm/aggregate_cnxt_standalone_compare.sbatch

#SBATCH --job-name=cnxt-standalone-compare
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

# Production ConvNeXt parquets (phase2)
PROD_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
# Cooldown ConvNeXt parquets (phase2v3)
COOLDOWN_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large_phase2v3"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
# multiscale_aggregate.py requires --model-path (only used for header sanity)
MODEL_PATH="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp10c_standalone_compare"
PROD_OUT="${OUT_DIR}/production"
COOL_OUT="${OUT_DIR}/cooldown"
mkdir -p "${PROD_OUT}" "${COOL_OUT}"
mkdir -p logs

# Sanity check: cooldown parquets must exist for scales 4, 5, 6, 7
for s in 4 5 6 7; do
    f="${COOLDOWN_PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing cooldown parquet at $f"
        exit 1
    fi
done
# Production parquets — scales 5, 6, 7 minimum (P scheme); 4 optional (K scheme)
for s in 5 6 7; do
    f="${PROD_PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing production parquet at $f"
        exit 1
    fi
done
HAS_PROD_S4=0
[ -f "${PROD_PARQUET_ROOT}/scale_4x4.parquet" ] && HAS_PROD_S4=1

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== ConvNeXt standalone: production vs cooldown ==="
echo "Production: ${PROD_PARQUET_ROOT} (scale 4: $([ $HAS_PROD_S4 -eq 1 ] && echo yes || echo no))"
echo "Cooldown:   ${COOLDOWN_PARQUET_ROOT}"
echo "Output:     ${OUT_DIR}"

# Schemes to test depending on what production has
if [ $HAS_PROD_S4 -eq 1 ]; then
    SCHEMES="K,P,I"
else
    SCHEMES="P,I"
    echo "(Production lacks scale 4 — skipping scheme K)"
fi

# ---- Production aggregation ----
echo ""
echo "--- Aggregating PRODUCTION ConvNeXt (${SCHEMES}) ---"
PROD_ARGS=""
[ $HAS_PROD_S4 -eq 1 ] && PROD_ARGS="${PROD_ARGS} --scale4-dir ${PROD_PARQUET_ROOT}/scale_4x4.parquet"
python "${SRC_ROOT}/multiscale_aggregate.py" \
    ${PROD_ARGS} \
    --scale5-dir "${PROD_PARQUET_ROOT}/scale_5x5.parquet" \
    --scale6-dir "${PROD_PARQUET_ROOT}/scale_6x6.parquet" \
    --scale7-dir "${PROD_PARQUET_ROOT}/scale_7x7.parquet" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${PROD_OUT}" \
    --temperature 1.5 \
    --alpha 0.30 \
    --topk 10 \
    --image-topk 64 \
    --schemes "${SCHEMES}" \
    --thresholds "0.075,0.085,0.100,0.125" \
    --csv-prefix "production_"

# ---- Cooldown aggregation ----
echo ""
echo "--- Aggregating COOLDOWN ConvNeXt (K,P,I) ---"
python "${SRC_ROOT}/multiscale_aggregate.py" \
    --scale4-dir "${COOLDOWN_PARQUET_ROOT}/scale_4x4.parquet" \
    --scale5-dir "${COOLDOWN_PARQUET_ROOT}/scale_5x5.parquet" \
    --scale6-dir "${COOLDOWN_PARQUET_ROOT}/scale_6x6.parquet" \
    --scale7-dir "${COOLDOWN_PARQUET_ROOT}/scale_7x7.parquet" \
    --model-path "${MODEL_PATH}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${COOL_OUT}" \
    --temperature 1.5 \
    --alpha 0.30 \
    --topk 10 \
    --image-topk 64 \
    --schemes "K,P,I" \
    --thresholds "0.075,0.085,0.100,0.125" \
    --csv-prefix "cooldown_"

echo ""
echo "=== Done ==="
echo ""
echo "Output CSVs:"
echo "  Production: ${PROD_OUT}"
echo "  Cooldown:   ${COOL_OUT}"
echo ""
echo "NEXT STEPS (laptop):"
echo "  1. Download both directories"
echo "  2. Run per-source proxy comparison:"
echo "       python C:/Users/alper/Desktop/analyze_exp16c.py \\"
echo "           production_scheme_P_T1.50_thr0.085_top10.csv \\"
echo "           cooldown_scheme_P_T1.50_thr0.085_top10.csv"
echo "     → diff function shows per-source +species/-species"
echo "  3. If cooldown looks structurally better (more alpine in alpine sources,"
echo "     less OOP in OPTMix), submit ONE LB on cooldown_scheme_P_T1.50_thr0.085_top10.csv"
echo "     to get ground-truth standalone F1."
echo "  4. Use the difference (cooldown_F1 - production_F1) as the gate:"
echo "     - If +0.005 to +0.015: continue to ensemble integration (Step 3 from earlier)"
echo "     - If flat or negative: stop; investigate before integrating"
