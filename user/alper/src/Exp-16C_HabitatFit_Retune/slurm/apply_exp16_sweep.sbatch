#!/bin/bash
# Exp-16C — apply + re-tune Exp-16 mechanisms on the new Exp-14C ViT-L pipeline.
#
# Path B (full re-tune) — 12 CSVs in one job:
#   Phase 1: locked Exp-16 verbatim (1 CSV) → measures transfer ratio
#   Phase 2: habitat-fit β re-tune (4 CSVs)
#   Phase 3: habitat-fit R re-tune (3 CSVs)
#   Phase 4: admission per-source variants (4 CSVs)
#
# Each variant calls Exp-16's apply_habitat_fit.py with the saved
# ensemble parquet from apply_exp14_mechanisms.sbatch as input.
#
# Reuses the production apply_habitat_fit.py — no new code.
#
# Usage:
#   sbatch slurm/apply_exp16_sweep.sbatch

#SBATCH --job-name=exp16c-sweep
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

# Configurable: which ensemble parquet to start from (default = Exp-14C locked)
ENS_PARQUET="${ENS_PARQUET:-${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms/exp14c_simw_b0.70_w0.88_thr0.085_w0.88_ensemble.parquet}"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"
ADMISSION_B="${EXP16_ROOT}/outputs/per_source/configs/B_OPTMix_n2.json"
ADMISSION_A="${EXP16_ROOT}/outputs/per_source/configs/A_combined.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_sweep"
mkdir -p "${OUT_DIR}"

# Sanity checks on inputs
for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}" "${ADMISSION_B}" "${ADMISSION_A}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"
        echo "If geo_profile / refined_source_ref / per_source configs are missing,"
        echo "upload from local repo (see Exp-16C_HabitatFit_Retune/docs/Exp16C_Plan.md)."
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C: full Exp-16 re-tune sweep on new ViT-L pipeline ==="
echo "Input parquet: ${ENS_PARQUET}"
echo "Output:        ${OUT_DIR}"

# Helper: run one habitat-fit variant
# Args: tag, beta, n_near_min, alt_flag, radius_km, ratio_floor, n_floor, per_source_json
run_variant() {
    local TAG="$1"; local BETA="$2"; local N_NEAR="$3"; local ALT="$4"
    local RADIUS="$5"; local R_FLOOR="$6"; local N_FLOOR="$7"; local PS_JSON="$8"
    local OUT_CSV="${OUT_DIR}/exp16c_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: β=${BETA} n=${N_NEAR} R=${RADIUS} ratio=${R_FLOOR} n_floor=${N_FLOOR} per_src=$(basename ${PS_JSON} 2>/dev/null || echo none) ---"
    local ALT_ARG=""
    [ "${ALT}" = "on" ] && ALT_ARG="--use-altitude"
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
        --n-near-min "${N_NEAR}" \
        ${ALT_ARG} \
        --threshold 0.085 \
        --topk 10 \
        --n-floor "${N_FLOOR}" \
        --ratio-floor "${R_FLOOR}" \
        ${PS_ARG} \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Phase 1: locked Exp-16 verbatim → measure transfer ratio
# ============================================================
run_variant "P1_locked"        0.25  30  on  200  0.5  3  "${ADMISSION_B}"

# ============================================================
# Phase 2: β sweep around 0.25
# ============================================================
run_variant "P2_b0.15"         0.15  30  on  200  0.5  3  "${ADMISSION_B}"
run_variant "P2_b0.20"         0.20  30  on  200  0.5  3  "${ADMISSION_B}"
run_variant "P2_b0.30"         0.30  30  on  200  0.5  3  "${ADMISSION_B}"
run_variant "P2_b0.40"         0.40  30  on  200  0.5  3  "${ADMISSION_B}"

# ============================================================
# Phase 3: R sweep at locked β
# ============================================================
run_variant "P3_R150"          0.25  30  on  150  0.5  3  "${ADMISSION_B}"
run_variant "P3_R250"          0.25  30  on  250  0.5  3  "${ADMISSION_B}"
run_variant "P3_R300"          0.25  30  on  300  0.5  3  "${ADMISSION_B}"

# ============================================================
# Phase 4: admission variants (test what transfers)
# ============================================================
# Phase A only — diagnostic for whether OPTMix n=2 still helps
run_variant "P4_phaseA_only"   0.25  30  on  200  0.5  3  "${ADMISSION_A}"
# Pure habitat-fit, no per-source admission overrides
run_variant "P4_no_per_src"    0.25  30  on  200  0.5  3  ""
# Tighter ratio_floor globally
run_variant "P4_r0.7_global"   0.25  30  on  200  0.7  3  "${ADMISSION_B}"
# Looser n_floor globally
run_variant "P4_n2_global"     0.25  30  on  200  0.5  2  "${ADMISSION_B}"

echo ""
echo "=== Done. 12 CSVs in ${OUT_DIR} ==="
echo ""
echo "Reference points:"
echo "  Exp-14C raw best (no Exp-16):      0.45728"
echo "  Old pipeline + full Exp-16:         0.49842 (target to beat)"
echo "  Predicted P1_locked (verbatim):     0.475-0.495 (transfer dependent)"
echo ""
echo "Submission strategy (~10 LB):"
echo "  Round 1 (1 LB): P1_locked → measures Exp-16 transfer ratio"
echo "  Round 2 (3-4 LB): P2 β variants → identify new β optimum"
echo "  Round 3 (2-3 LB): P3 R variants if β unchanged, OR P4 admission variants"
echo "  Total: ~10 LB submissions"
