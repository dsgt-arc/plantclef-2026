#!/bin/bash
# Exp-16C Stage 4 — combine Stage 3 independent per-source winners.
#
# Stage 3 single-source wins (vs no-per-src baseline 0.48830 at β=0.10):
#   OPTMix n_floor=2:  +0.00269  → 0.49099
#   LISAH r=0.6:       +0.00054  → 0.48884
#   LISAH r=0.7:       +0.00054  → 0.48884 (tied with r=0.6)
#   LISAH n_floor=2:   +0.00108  → 0.48938
# RNNB and GUARDEN show no admission tuning room (rules don't fire).
#
# CRITICAL: OPTMix r=0.3 (old win) HURTS (−0.00483) on new pipeline. DROP IT.
# Old B_OPTMix_n2 = LISAH r=0.6 + OPTMix r=0.3 + OPTMix n=2 → contains a hurt.
#
# Stage 4 tests combinations of the actual single-source winners.
# Held: β=0.10, R=200, n_near=30, alt-on, thr=0.085, k=10, defaults r=0.5 n=3.
#
# Total: 8 CSVs.
#
# Usage:
#   sbatch slurm/stage4_combine_winners.sbatch

#SBATCH --job-name=exp16c-s4-combine
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage4"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"; exit 1
    fi
done

# Generate combination configs
cat > "${PS_DIR}/C1_OPTMix_n2_LISAH_r0.6.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.6}}
EOF
cat > "${PS_DIR}/C2_OPTMix_n2_LISAH_r0.7.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.7}}
EOF
cat > "${PS_DIR}/C3_OPTMix_n2_LISAH_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/C4_OPTMix_n2_LISAH_r0.6_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.6, "n_floor": 2}}
EOF
cat > "${PS_DIR}/C5_OPTMix_n2_LISAH_r0.7_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.7, "n_floor": 2}}
EOF
# Global n=2 — applied to all count-bound sources
cat > "${PS_DIR}/C6_global_n2.json" <<'EOF'
{"RNNB": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "OPTMix": {"n_floor": 2}, "GUARDEN": {"n_floor": 2}}
EOF
# Best combo + RNNB n=2 (in case it composes positively despite being flat alone)
cat > "${PS_DIR}/C7_best_plus_RNNB_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.6, "n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
# Reference comparison: re-run old B_OPTMix_n2 to confirm 0.48992 baseline
cat > "${PS_DIR}/C8_oldB_OPTMix_n2.json" <<'EOF'
{"LISAH": {"ratio_floor": 0.6}, "OPTMix": {"ratio_floor": 0.3, "n_floor": 2}}
EOF

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 4: combine independent per-source winners (8 CSVs) ==="
echo "Held: β=0.10, R=200, n_near=30, alt-on, thr=0.085, k=10"

# Helper
run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s4_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: $(cat ${PS_JSON_PATH}) ---"
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
        --n-near-min 30 \
        --use-altitude \
        --threshold 0.085 \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${PS_JSON_PATH}" \
        --out-csv "${OUT_CSV}"
}

run_v "C1_OPTMix_n2_LISAH_r0.6"        "${PS_DIR}/C1_OPTMix_n2_LISAH_r0.6.json"
run_v "C2_OPTMix_n2_LISAH_r0.7"        "${PS_DIR}/C2_OPTMix_n2_LISAH_r0.7.json"
run_v "C3_OPTMix_n2_LISAH_n2"          "${PS_DIR}/C3_OPTMix_n2_LISAH_n2.json"
run_v "C4_OPTMix_n2_LISAH_r0.6_n2"     "${PS_DIR}/C4_OPTMix_n2_LISAH_r0.6_n2.json"
run_v "C5_OPTMix_n2_LISAH_r0.7_n2"     "${PS_DIR}/C5_OPTMix_n2_LISAH_r0.7_n2.json"
run_v "C6_global_n2"                   "${PS_DIR}/C6_global_n2.json"
run_v "C7_best_plus_RNNB_n2"           "${PS_DIR}/C7_best_plus_RNNB_n2.json"
run_v "C8_oldB_OPTMix_n2"              "${PS_DIR}/C8_oldB_OPTMix_n2.json"

echo ""
echo "=== Done. 8 CSVs in ${OUT_DIR} ==="
echo ""
echo "Predicted F1 (assuming additive composition of Stage 3 wins):"
echo "  C1 (OPTMix n2 + LISAH r0.6):    ~0.49153 (+0.00269 + 0.00054)"
echo "  C2 (OPTMix n2 + LISAH r0.7):    ~0.49153 (same)"
echo "  C3 (OPTMix n2 + LISAH n2):      ~0.49207 (+0.00269 + 0.00108)"
echo "  C4 (OPTMix n2 + LISAH r0.6+n2): ~0.49261 (+0.00269 + 0.00054 + 0.00108)"
echo "  C5 (OPTMix n2 + LISAH r0.7+n2): ~0.49261 (same)"
echo "  C6 (global n2):                 ~0.491-0.493 (could regress if n=2 hurts inactive sources)"
echo "  C7 (C4 + RNNB n2):              ~0.49261 (RNNB additive ~ 0)"
echo "  C8 (old B_OPTMix_n2 sanity):    0.48992 (verifies old measurement)"
echo ""
echo "Submission strategy (~5 LB):"
echo "  Submit C4 first — most likely best by additivity"
echo "  Then C1, C3 to test individual contributions"
echo "  C8 as sanity check vs the locked 0.48992"
echo ""
echo "After Stage 4 lock, remaining gap to 0.49896:"
echo "  ~0.006 → addressed by Stage 5 (threshold + topk + n_near sweeps at new lock)"
