#!/bin/bash
# Exp-14C Stage 1ext-b: ensemble sweep extending Stage 1's parameter grid in
# the directions Stage 1 indicated were under-explored.
#
# Stage 1 patterns:
#   - Lower beta wins monotonically (0.75 best of {0.75, 0.85, 0.95})
#   - Higher threshold wins monotonically (0.090 > 0.075)
#   - w=0.80 best, w=0.75 close
#
# REVISED 2026-05-03: standalone phase2v3 K optimization landed at thr=0.085
# (alpha=0.30 win = 0.41186). Standalone curve at alpha=0.20 was sharp:
#   thr 0.080→0.40003 / 0.085→0.41097 / 0.090→0.40951 / 0.095→0.41037 / 0.100→0.40230
# That collapses the previous "higher threshold" intuition — Stage 1 only saw
# 0.090 > 0.075 because we missed 0.085. Ensemble likely peaks 0.080-0.100,
# NOT 0.100-0.130 (which the old sweep was extending toward).
#
# Stage 1ext sweep (revised):
#   beta_multi  ∈ {0.55, 0.60, 0.65, 0.70, 0.75}    (4 NEW + Stage 1's best)
#   weight w    ∈ {0.77, 0.80, 0.83}               (narrowed around peak)
#   threshold   ∈ {0.075, 0.085, 0.095, 0.100, 0.115}  (centered on 0.085 with cushion)
# Total: 5 × 3 × 5 = 75 CSVs.
#
# Don't submit all 60 to LB — pick a representative subset (e.g., diagonal
# scan along the peak ridge) for the first round, then expand.
#
# Prerequisite: build_prefusion_extended.sbatch must have completed
# (provides b={0.55, 0.60, 0.65, 0.70} parquets). The b=0.75 parquet should
# already exist from Stage 1.

#SBATCH --job-name=exp14c-ens-stage1ext
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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
OUT_BASE="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_stage1ext"

mkdir -p "${OUT_BASE}"

# Sanity checks
for b in 0.55 0.60 0.65 0.70 0.75; do
    f="${PREFUSION_DIR}/vitL_prefusion_b${b}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing prefusion parquet at $f"
        echo "Run build_prefusion_extended.sbatch first (and Stage 1 build for b=0.75)."
        exit 1
    fi
done
for s in 4 5 6 7; do
    f="${CNXT_PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing ConvNeXt scale ${s}x${s} at $f"
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C Stage 1ext-b: ensemble sweep (5 betas × 3 weights × 5 thresholds = 75 CSVs) ==="
echo "Threshold range revised based on standalone result: K, T=1.5, alpha=0.30, thr=0.085 = 0.41186"

for BETA in 0.55 0.60 0.65 0.70 0.75; do
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
        --weights "0.77,0.80,0.83" \
        --thresholds "0.075,0.085,0.095,0.100,0.115" \
        --topk 10 \
        --vit-aggregator max \
        --cnxt-aggregator max \
        --tag "exp14c_b${BETA}"

    # Tag CSVs so they don't collide on download
    for f in "${OUT_DIR}"/exp14c_b${BETA}_*.csv; do
        [ -e "$f" ] || continue
        base="$(basename "$f" .csv)"
        case "$base" in
            *_vitLK_cnxtK) ;;
            *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtK.csv" ;;
        esac
    done

    # Clean up leftover intermediate parquets
    rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true
done

echo ""
echo "=== Stage 1ext-b done ==="
echo "All 75 CSVs under ${OUT_BASE}/b{0.55,0.60,0.65,0.70,0.75}/"
echo ""
echo "Submission strategy (~12 LB submissions):"
echo "  Round 1 (5 LB): one CSV per beta at (w=0.80, thr=0.085 — standalone optimum)"
echo "                  → identifies winning beta"
echo "  Round 2 (4 LB): at winning beta, sweep thr {0.075, 0.095, 0.100, 0.115} at w=0.80"
echo "                  → confirms threshold optimum"
echo "  Round 3 (3 LB): at winning (beta, thr), sweep w {0.77, 0.83}"
echo "                  → fine-tunes weight"
echo ""
echo "Decision rules (same as Stage 1):"
echo "  - best CSV >= 0.456: success, proceed to Exp-14D (stack Exp-16 mechanisms)"
echo "  - best CSV in [0.444, 0.456): marginal — proceed but lower expectations"
echo "  - best CSV  < 0.444: real regression — diagnose retriever/classifier integration"
