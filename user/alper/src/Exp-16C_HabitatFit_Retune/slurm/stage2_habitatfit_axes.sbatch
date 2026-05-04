#!/bin/bash
# Exp-16C Stage 2 — proper independent-axis sweep of habitat-fit on the new pipeline.
#
# DESIGN PRINCIPLE: do NOT inherit the old pipeline's locked admission config
# (B_OPTMix_n2). Sweep habitat-fit axes IN ISOLATION first — pure habitat-fit,
# no per-source overrides. Then Stage 3 will add per-source admission tuning
# from scratch on top of the new locked habitat-fit point.
#
# Axes swept (independent):
#   β: {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}  at R=200, n=30
#   R: depends on Stage 2A winner (skipped here, run as Stage 2B after Stage 2A submits)
#   n_near_min: depends on Stage 2A winner (skipped here, run as Stage 2C)
#
# This script does Stage 2A only (β sweep at R=200, n=30, no per-source).
# Stage 2B (R sweep) and Stage 2C (n_near sweep) are separate sbatches written
# AFTER Stage 2A's β winner is identified via LB.
#
# Total: 7 CSVs (β values).
#
# Usage:
#   sbatch slurm/stage2_habitatfit_axes.sbatch

#SBATCH --job-name=exp16c-s2A-beta
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
EXP16_ROOT="${REPO_ROOT}/user/alper/src/Exp-16_Multi_Species_Discovery"
EXP16_SCRIPTS="${EXP16_ROOT}/scripts"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

ENS_PARQUET="${ENS_PARQUET:-${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms/exp14c_simw_b0.70_w0.88_thr0.085_w0.88_ensemble.parquet}"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage2A_beta"
mkdir -p "${OUT_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 2A: β sweep, NO per-source admission ==="
echo "Held: R=200, n_near=30, alt-on, thr=0.085, k=10, global N3_r0.5 admission"
echo "Output: ${OUT_DIR}"

# Helper — runs habitat-fit + global N3_r0.5 admission only (no --per-source-json)
run_beta() {
    local BETA="$1"
    local OUT_CSV="${OUT_DIR}/exp16c_s2A_b${BETA}.csv"
    echo ""
    echo "--- β=${BETA} ---"
    python "${EXP16_SCRIPTS}/apply_habitat_fit.py" \
        --ensemble-parquet "${ENS_PARQUET}" \
        --geo-profile "${GEO_PROFILE}" \
        --species-map "${SPECIES_MAP}" \
        --geo-mask "${GEO_MASK}" \
        --habitat-mask "${HABITAT_MASK}" \
        --biogeo-npz "${BIOGEO_NPZ}" \
        --refined-ref-json "${REFINED_REF}" \
        --radius-km 200 \
        --beta "${BETA}" \
        --n-near-min 30 \
        --use-altitude \
        --threshold 0.085 \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --out-csv "${OUT_CSV}"
}

for BETA in 0.05 0.10 0.15 0.20 0.25 0.30 0.40; do
    run_beta "${BETA}"
done

echo ""
echo "=== Stage 2A done. 7 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy:"
echo "  Submit all 7 to LB (cheap — they tell us the true β response on the new pipeline,"
echo "  isolated from per-source admission contamination)."
echo ""
echo "Next stages (write after Stage 2A LB results):"
echo "  Stage 2B: R sweep at locked β*, n=30 (6 values)"
echo "  Stage 2C: n_near_min sweep at locked (β*, R*) (7 values)"
echo "  Stage 3:  per-source admission sweep on top of Stage 2 winner"
