#!/bin/bash
# Exp-16C Stage 2B — exploit the β × admission interaction discovered in Stage 2A.
#
# Stage 2A finding (no per-source admission):
#   β=0.10 = 0.48830 (true peak), β=0.20 = 0.48803, β=0.05 = 0.48605
# Stage 1 finding (with B_OPTMix_n2 admission):
#   β=0.20 + B_OPTMix_n2 = 0.48911 (current best)
#   β=0.25 + B_OPTMix_n2 = 0.48298 (per-source HURTS at β=0.25)
#
# Hypothesis: per-source admission adds +0.001 per 0.05 of β reduction
# (more aggressive demotion → more holes for admission to fill helpfully).
#
# Test: at β=0.10, β=0.12, β=0.08 — does per-source admission keep helping?
# Also test β=0.05 + per-source to confirm β floor.
#
# Plus: R sweep at the new candidate β=0.10 (no per-source for clean signal).
#
# Total: 12 CSVs.
#
# Usage:
#   sbatch slurm/stage2B_beta_admission_interaction.sbatch

#SBATCH --job-name=exp16c-s2B-interact
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
ADMISSION_B="${EXP16_ROOT}/outputs/per_source/configs/B_OPTMix_n2.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage2B"
mkdir -p "${OUT_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}" "${ADMISSION_B}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"; exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 2B: β × admission interaction + R sweep at new β ==="

run_v() {
    local TAG="$1"; local BETA="$2"; local RADIUS="$3"; local PS_JSON="$4"
    local OUT_CSV="${OUT_DIR}/exp16c_s2B_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: β=${BETA} R=${RADIUS} per_src=$(basename ${PS_JSON} 2>/dev/null || echo none) ---"
    local PS_ARG=""
    [ -n "${PS_JSON}" ] && PS_ARG="--per-source-json ${PS_JSON}"
    python "${EXP16_SCRIPTS}/apply_habitat_fit.py" \
        --ensemble-parquet "${ENS_PARQUET}" \
        --geo-profile "${GEO_PROFILE}" \
        --species-map "${SPECIES_MAP}" \
        --geo-mask "${GEO_MASK}" \
        --habitat-mask "${HABITAT_MASK}" \
        --biogeo-npz "${BIOGEO_NPZ}" \
        --refined-ref-json "${REFINED_REF}" \
        --radius-km "${RADIUS}" \
        --beta "${BETA}" \
        --n-near-min 30 \
        --use-altitude \
        --threshold 0.085 \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        ${PS_ARG} \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Family I: β × admission interaction near the new peak
# Held: R=200, n=30, alt-on, thr=0.085
# ============================================================
# β=0.05 with admission (was 0.48605 without)
run_v "I_b0.05_admB"     0.05  200  "${ADMISSION_B}"
# β=0.08 (interpolation between 0.05 and 0.10) — both with and without admission
run_v "I_b0.08_noPS"     0.08  200  ""
run_v "I_b0.08_admB"     0.08  200  "${ADMISSION_B}"
# β=0.10 with admission — KEY TEST (no-PS gave 0.48830)
run_v "I_b0.10_admB"     0.10  200  "${ADMISSION_B}"
# β=0.12 fine — both
run_v "I_b0.12_noPS"     0.12  200  ""
run_v "I_b0.12_admB"     0.12  200  "${ADMISSION_B}"
# β=0.15 with admission (no-PS gave 0.48661)
run_v "I_b0.15_admB"     0.15  200  "${ADMISSION_B}"

# ============================================================
# Family R: R sweep at NEW peak β=0.10 (no per-source for clean signal)
# Held: β=0.10, n=30, alt-on, thr=0.085
# ============================================================
run_v "R_R150_b0.10"     0.10  150  ""
run_v "R_R250_b0.10"     0.10  250  ""
run_v "R_R300_b0.10"     0.10  300  ""

# ============================================================
# Family Rb: R sweep at β=0.10 with admission (if admission helps at β=0.10)
# ============================================================
run_v "Rb_R250_b0.10_adm" 0.10  250  "${ADMISSION_B}"
run_v "Rb_R150_b0.10_adm" 0.10  150  "${ADMISSION_B}"

echo ""
echo "=== Done. 12 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~8-10 LB):"
echo "  Round 1 (4 LB): I_b0.05_admB, I_b0.08_admB, I_b0.10_admB, I_b0.12_admB"
echo "    → confirms β optimum WITH admission. Compare to current best (β=0.20+adm=0.48911)"
echo "    → if any > 0.491, lock new (β, admission) point and proceed to Stage 3 from there"
echo "  Round 2 (3 LB): R_R150/250/300_b0.10 — find R optimum at new β"
echo "  Round 3 (3 LB): Rb_R150/250 + the missing no-PS β values for clean curves"
echo ""
echo "Decision targets (cumulative):"
echo "  Best Stage 2B + Stage 3 per-source = expected 0.495-0.500"
echo "  Target to beat: 0.49896"
