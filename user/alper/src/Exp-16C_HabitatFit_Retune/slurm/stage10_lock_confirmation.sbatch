#!/bin/bash
# Exp-16C Stage 10 — final lock confirmation at β=0.15.
#
# Stage 9 found β=0.15 → 0.50488 (best so far). β curve is symmetric:
#   β=0.075: 0.50137, β=0.10: 0.50304, β=0.15: 0.50488, β=0.20: 0.50088.
# R is locked hard at 200 (Stage 9: R=150 and R=250+ all collapse by ~0.02).
#
# This stage confirms n_near and threshold optima haven't shifted under
# the new β=0.15 anchor. If A_b0.15 reproduces 0.50488 and the side-points
# don't beat it, the lock is final.
#
# Held: alt-on, β=0.15, R=200, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 (Stage 4 lock)
#
# Sweeps:
#   A) n_near at thr=0.085  ∈ {35, 45}                  (2 variants)
#   B) thr at n=40          ∈ {0.080, 0.090}            (2 variants)
#   E) Anchor reproduction (n=40, thr=0.085)             (1 variant)
#
# Total: 5 CSVs.
#
# Usage:
#   sbatch slurm/stage10_lock_confirmation.sbatch

#SBATCH --job-name=exp16c-s10-lock
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage10"
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
echo "=== Exp-16C Stage 10: lock confirmation at β=0.15 (5 CSVs) ==="
echo "Lock: alt-on, β=0.15, R=200, k=10, OPTMix n=2 + LISAH n=2"
echo "Anchor expected: 0.50488 (Stage 9 A_b0.15 reproduction)"

run_v() {
    local TAG="$1"; local THR="$2"; local N_NEAR="$3"
    local OUT_CSV="${OUT_DIR}/exp16c_s10_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: thr=${THR} n_near=${N_NEAR} ---"
    python "${EXP16_SCRIPTS}/apply_habitat_fit.py" \
        --ensemble-parquet "${ENS_PARQUET}" \
        --geo-profile "${GEO_PROFILE}" \
        --species-map "${SPECIES_MAP}" \
        --geo-mask "${GEO_MASK}" \
        --habitat-mask "${HABITAT_MASK}" \
        --biogeo-npz "${BIOGEO_NPZ}" \
        --refined-ref-json "${REFINED_REF}" \
        --radius-km 200 \
        --beta 0.15 \
        --n-near-min "${N_NEAR}" \
        --use-altitude \
        --threshold "${THR}" \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${LOCKED_PS}" \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Family A: n_near side-points at thr=0.085
# ============================================================
run_v "A_n35"      0.085  35
run_v "A_n45"      0.085  45

# ============================================================
# Family B: threshold side-points at n=40
# ============================================================
run_v "B_thr0.080" 0.080  40
run_v "B_thr0.090" 0.090  40

# ============================================================
# Family E: anchor reproduction (sanity)
# ============================================================
run_v "E_anchor"   0.085  40

echo ""
echo "=== Stage 10 done. 5 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~3-4 LB):"
echo ""
echo "Round 1 (1 LB) — anchor sanity:"
echo "  E_anchor → must reproduce 0.50488 to validate the pipeline"
echo ""
echo "Round 2 (2-3 LB) — side-points:"
echo "  A_n35, A_n45 (if n_near optimum shifted under β=0.15)"
echo "  B_thr0.080, B_thr0.090 (if threshold optimum shifted)"
echo ""
echo "Decision: if E_anchor = 0.50488 and no side-point beats it, the lock is final."
echo "Final config: β=0.15, R=200, n=40, thr=0.085, k=10, OPTMix n=2 + LISAH n=2"
