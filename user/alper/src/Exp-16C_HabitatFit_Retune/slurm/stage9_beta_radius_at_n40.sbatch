#!/bin/bash
# Exp-16C Stage 9 — β × R re-sweep at new n_near=40 lock.
#
# Stage 8 confirmed thr=0.085 and k=10 are still optimal at n=40
# (curve symmetric around old peak; no shift from Stage 5).
#
# Stage 2A's β sweep was anchored at n=30. Widening the spatial
# neighborhood from 30→40 changes how the geo prior is averaged:
# more neighbors → smoother prior → demotion strength may want to scale up.
# Similarly, R=200 may not be optimal at the new n=40 lock.
#
# Held: alt-on, n_near_min=40, thr=0.085, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 (Stage 4 lock)
#
# Sweeps:
#   A) β at R=200    ∈ {0.05, 0.075, 0.15, 0.20, 0.30, 0.50}    (6 variants)
#   B) R at β=0.10   ∈ {100, 150, 250, 300, 400}                (5 variants)
#   C) β × R cross   {β=0.15, R=300}, {β=0.20, R=300},
#                    {β=0.15, R=150}, {β=0.05, R=300}            (4 variants)
#   E) Anchor reproduction (β=0.10, R=200)                       (1 variant)
#
# Total: 16 CSVs.
#
# Usage:
#   sbatch slurm/stage9_beta_radius_at_n40.sbatch

#SBATCH --job-name=exp16c-s9-betaR
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage9"
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
echo "=== Exp-16C Stage 9: β × R re-sweep at n_near=40 (16 CSVs) ==="
echo "Lock: alt-on, n=40, thr=0.085, k=10, OPTMix n=2 + LISAH n=2"
echo "Stage 7 anchor: 0.50304 (β=0.10, R=200)"

run_v() {
    local TAG="$1"; local BETA="$2"; local R="$3"
    local OUT_CSV="${OUT_DIR}/exp16c_s9_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: β=${BETA} R=${R} ---"
    python "${EXP16_SCRIPTS}/apply_habitat_fit.py" \
        --ensemble-parquet "${ENS_PARQUET}" \
        --geo-profile "${GEO_PROFILE}" \
        --species-map "${SPECIES_MAP}" \
        --geo-mask "${GEO_MASK}" \
        --habitat-mask "${HABITAT_MASK}" \
        --biogeo-npz "${BIOGEO_NPZ}" \
        --refined-ref-json "${REFINED_REF}" \
        --radius-km "${R}" \
        --beta "${BETA}" \
        --n-near-min 40 \
        --use-altitude \
        --threshold 0.085 \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${LOCKED_PS}" \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Family A: β sweep at R=200, n=40
# ============================================================
run_v "A_b0.05"     0.05   200
run_v "A_b0.075"    0.075  200
run_v "A_b0.15"     0.15   200
run_v "A_b0.20"     0.20   200
run_v "A_b0.30"     0.30   200
run_v "A_b0.50"     0.50   200

# ============================================================
# Family B: R sweep at β=0.10, n=40
# ============================================================
run_v "B_R100"      0.10   100
run_v "B_R150"      0.10   150
run_v "B_R250"      0.10   250
run_v "B_R300"      0.10   300
run_v "B_R400"      0.10   400

# ============================================================
# Family C: β × R cross-points (chase smoother-prior + stronger-demotion combos)
# ============================================================
run_v "C_b0.15_R300"  0.15  300
run_v "C_b0.20_R300"  0.20  300
run_v "C_b0.15_R150"  0.15  150
run_v "C_b0.05_R300"  0.05  300

# ============================================================
# Family E: anchor reproduction
# ============================================================
run_v "E_anchor"     0.10   200

echo ""
echo "=== Stage 9 done. 16 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~6-8 LB):"
echo ""
echo "Round 1 (4 LB) — β around current anchor:"
echo "  A_b0.075, A_b0.15, A_b0.20, A_b0.30"
echo "  → does the demotion-strength optimum shift up under wider neighborhood?"
echo ""
echo "Round 2 (3 LB) — R sweep:"
echo "  B_R150, B_R300, B_R400"
echo "  → wider radius averages over more diverse training data"
echo ""
echo "Round 3 (1-2 LB) — best-cross combo:"
echo "  whichever β × R combo from Family C looks most promising after Rounds 1-2"
echo ""
echo "Current peak: 0.50304. Stretch: 0.51+."
