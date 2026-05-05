#!/bin/bash
# Exp-16C Stage 8 — threshold + topk re-sweep at new n_near=40 lock.
#
# Stage 7 confirmed n_near_min=40 → F1=0.50304 (peak). Stage 5's threshold
# and topk landscapes were anchored at n=30; widening the spatial neighborhood
# may have shifted those optima.
#
# Held: β=0.10, R=200, alt-on, n_near_min=40
#       Per-source admission: OPTMix n=2 + LISAH n=2 (Stage 4 lock)
#
# Sweeps:
#   A) Threshold ∈ {0.070, 0.075, 0.080, 0.090, 0.095, 0.100}  (6 variants)
#   B) Topk ∈ {12, 15, 20}                                      (3 variants)
#   E) Anchor reproduction (n=40 baseline)                       (1 variant)
#
# Total: 10 CSVs.
#
# Usage:
#   sbatch slurm/stage8_thr_topk_at_n40.sbatch

#SBATCH --job-name=exp16c-s8-n40refine
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage8"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

cat > "${PS_DIR}/locked.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
LOCKED_PS="${PS_DIR}/locked.json"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing required file: $f"; exit 1; fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 8: thr+topk re-sweep at n_near=40 (10 CSVs) ==="
echo "Lock: β=0.10, R=200, alt-on, n_near=40, OPTMix n=2 + LISAH n=2"
echo "Stage 7 anchor: 0.50304 (n=40, thr=0.085, k=10)"

run_v() {
    local TAG="$1"; local THR="$2"; local TOPK="$3"
    local OUT_CSV="${OUT_DIR}/exp16c_s8_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: thr=${THR} k=${TOPK} ---"
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
        --n-near-min 40 \
        --use-altitude \
        --threshold "${THR}" \
        --topk "${TOPK}" \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${LOCKED_PS}" \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Family A: threshold sweep at n=40 lock (k=10)
# ============================================================
run_v "A_thr0.070"  0.070  10
run_v "A_thr0.075"  0.075  10
run_v "A_thr0.080"  0.080  10
run_v "A_thr0.090"  0.090  10
run_v "A_thr0.095"  0.095  10
run_v "A_thr0.100"  0.100  10

# ============================================================
# Family B: topk sweep at n=40 lock (thr=0.085)
# ============================================================
run_v "B_k12"       0.085  12
run_v "B_k15"       0.085  15
run_v "B_k20"       0.085  20

# ============================================================
# Family E: anchor reproduction (sanity check)
# ============================================================
run_v "E_anchor"    0.085  10

echo ""
echo "=== Stage 8 done. 10 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~5-7 LB):"
echo ""
echo "Round 1 (4 LB) — threshold around old peak:"
echo "  A_thr0.080, A_thr0.090, A_thr0.075, A_thr0.095"
echo "  → does the threshold optimum shift under the wider spatial neighborhood?"
echo ""
echo "Round 2 (2 LB) — topk:"
echo "  B_k12, B_k15"
echo "  → at n=40, alpine saturation pattern may have changed; cap may now help"
echo ""
echo "Round 3 (1 LB):"
echo "  E_anchor as sanity check, should reproduce 0.50304"
echo ""
echo "Current peak: 0.50304. Target was 0.49896 (cleared by +0.00408)."
echo "Stretch goal: 0.51+ if threshold or topk shifts under n=40 anchor."
