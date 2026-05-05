#!/bin/bash
# Exp-16C Stage 5 — sweep untested dimensions at Stage 4 locked config.
#
# Locked at Stage 4: β=0.10, R=200, n_near=30, alt-on, thr=0.085, k=10
#                    + per-source admission: OPTMix n=2, LISAH n=2
# F1 = 0.49207. Target: beat 0.49896. Gap: 0.00689.
#
# Untested dimensions at the new locked admission point:
#   A) Threshold (currently 0.085 — never re-tuned at full Exp-16 stack)
#   B) topk (currently 10 — old pipeline confirmed; verify on new)
#   C) n_near_min (currently 30 — old pipeline optimum; new pipeline untested)
#   D) Altitude gate ablation (currently on)
#
# Total: 19 CSVs.
#
# Usage:
#   sbatch slurm/stage5_threshold_topk_n_near.sbatch

#SBATCH --job-name=exp16c-s5-finals
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage5"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

# Stage 4 locked admission: OPTMix n=2, LISAH n=2
cat > "${PS_DIR}/locked.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
LOCKED_PS="${PS_DIR}/locked.json"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"; exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 5: threshold + topk + n_near + altitude (19 CSVs) ==="
echo "Locked: β=0.10, R=200, alt-on, OPTMix n=2 + LISAH n=2"
echo "Stage 4 anchor F1: 0.49207"

# Helper
run_v() {
    local TAG="$1"; local THR="$2"; local TOPK="$3"; local N_NEAR="$4"; local ALT="$5"
    local OUT_CSV="${OUT_DIR}/exp16c_s5_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: thr=${THR} k=${TOPK} n_near=${N_NEAR} alt=${ALT} ---"
    local ALT_ARG=""
    [ "${ALT}" = "on" ] && ALT_ARG="--use-altitude"
    python "${EXP16_SCRIPTS}/apply_habitat_fit.py" \
        --ensemble-parquet "${ENS_PARQUET}" \
        --geo-profile "${GEO_PROFILE}" \
        --species-map "${SPECIES_MAP}" \
        --geo-mask "${GEO_MASK}" \
        --habitat-mask "${HABITAT_MASK}" \
        --biogeo-npz "${BIOGEO_NPZ}" \
        --refined-ref-json "${REFINED_REF}" \
        --radius-km 200 \
        --beta 0.10 \
        --n-near-min "${N_NEAR}" \
        ${ALT_ARG} \
        --threshold "${THR}" \
        --topk "${TOPK}" \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${LOCKED_PS}" \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Family A: threshold sweep at locked config (k=10, n_near=30, alt-on)
# ============================================================
run_v "A_thr0.060"  0.060  10  30  on
run_v "A_thr0.070"  0.070  10  30  on
run_v "A_thr0.075"  0.075  10  30  on
run_v "A_thr0.080"  0.080  10  30  on
run_v "A_thr0.090"  0.090  10  30  on
run_v "A_thr0.095"  0.095  10  30  on
run_v "A_thr0.100"  0.100  10  30  on
run_v "A_thr0.110"  0.110  10  30  on
run_v "A_thr0.120"  0.120  10  30  on
run_v "A_thr0.130"  0.130  10  30  on
run_v "A_thr0.150"  0.150  10  30  on

# ============================================================
# Family B: topk variations (thr=0.085, n_near=30, alt-on)
# ============================================================
run_v "B_k12"       0.085  12  30  on
run_v "B_k15"       0.085  15  30  on
run_v "B_k20"       0.085  20  30  on

# ============================================================
# Family C: n_near_min sweep (thr=0.085, k=10, alt-on)
# ============================================================
run_v "C_n15"       0.085  10  15  on
run_v "C_n20"       0.085  10  20  on
run_v "C_n25"       0.085  10  25  on
run_v "C_n40"       0.085  10  40  on
run_v "C_n50"       0.085  10  50  on

# ============================================================
# Family D: altitude ablation (thr=0.085, k=10, n=30)
# ============================================================
run_v "D_alt_off"   0.085  10  30  off

# ============================================================
# Family E: anchor reproduction (verifies pipeline)
# ============================================================
run_v "E_anchor"    0.085  10  30  on

echo ""
echo "=== Stage 5 done. 19 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~10-12 LB):"
echo ""
echo "Round 1 (5 LB) — threshold (highest expected upside):"
echo "  A_thr0.075, A_thr0.080, A_thr0.090, A_thr0.095, A_thr0.100"
echo "  → identifies threshold optimum at full Exp-16 stack"
echo ""
echo "Round 2 (3 LB) — n_near + topk:"
echo "  C_n20, C_n40, B_k12 — most likely n_near and topk variants"
echo ""
echo "Round 3 (2-3 LB):"
echo "  D_alt_off (ablation), and follow-up if any A/C/B winner suggests next test"
echo ""
echo "Predicted ceiling: 0.495-0.500. Target 0.49896."
echo "Plus E_anchor as sanity check should reproduce 0.49207."
