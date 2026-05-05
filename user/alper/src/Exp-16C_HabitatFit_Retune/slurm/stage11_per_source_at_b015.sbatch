#!/bin/bash
# Exp-16C Stage 11 — per-source admission re-tune at β=0.15, n=40 lock.
#
# Stages 3/4 found the per-source admission winners at β=0.10, n=30:
#   OPTMix n_floor=2 (+0.00269), LISAH n_floor=2 (+0.00108)
#   OPTMix r=0.3 HURT (−0.00483) — dropped
#   RNNB and GUARDEN flat
#   LISAH r=0.6/0.7 marginal (+0.00054, tied)
#
# Under the new anchor (β=0.15, n=40), the geo prior is averaged over
# more neighbors and demoted harder. This may shift admission optima:
#   - Sources with pulled-down confidence may want LOOSER admission
#   - Sources with marginal predictions may want TIGHTER admission
#   - OPTMix r=0.3 may no longer hurt (geo demotion does the work now)
#
# Held: alt-on, β=0.15, R=200, n_near_min=40, thr=0.085, k=10
#       Carried forward: OPTMix n_floor=2 + LISAH n_floor=2
#
# Sweeps (ALL variants STACK on the carried-forward OPTMix n=2 + LISAH n=2):
#   A) OPTMix n_floor refinement      ∈ {1, 3, 4}            (3 variants)
#   B) LISAH n_floor refinement       ∈ {1, 3, 4}            (3 variants)
#   C) LISAH ratio_floor stack        ∈ {0.6, 0.7, 0.8}      (3 variants)
#   D) OPTMix ratio_floor stack       ∈ {0.3, 0.4, 0.6}      (3 variants)
#   E) RNNB / GUARDEN re-test         RNNB n=2, GUARDEN n=2  (2 variants)
#   F) Anchor reproduction (carried-forward only)             (1 variant)
#
# Total: 15 CSVs.
#
# Usage:
#   sbatch slurm/stage11_per_source_at_b015.sbatch

#SBATCH --job-name=exp16c-s11-persrc
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage11"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing required file: $f"; exit 1; fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 11: per-source admission re-tune at β=0.15 (15 CSVs) ==="
echo "Lock: alt-on, β=0.15, R=200, n=40, thr=0.085, k=10"
echo "Anchor (carried-forward OPTMix n=2 + LISAH n=2): 0.50488"

run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s11_${TAG}.csv"
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
        --beta 0.15 \
        --n-near-min 40 \
        --use-altitude \
        --threshold 0.085 \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${PS_JSON_PATH}" \
        --out-csv "${OUT_CSV}"
}

# ============================================================
# Family A: OPTMix n_floor refinement (LISAH stays at n=2)
# ============================================================
cat > "${PS_DIR}/A_OPTMix_n1.json" <<'EOF'
{"OPTMix": {"n_floor": 1}, "LISAH": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/A_OPTMix_n3.json" <<'EOF'
{"OPTMix": {"n_floor": 3}, "LISAH": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/A_OPTMix_n4.json" <<'EOF'
{"OPTMix": {"n_floor": 4}, "LISAH": {"n_floor": 2}}
EOF
run_v "A_OPTMix_n1" "${PS_DIR}/A_OPTMix_n1.json"
run_v "A_OPTMix_n3" "${PS_DIR}/A_OPTMix_n3.json"
run_v "A_OPTMix_n4" "${PS_DIR}/A_OPTMix_n4.json"

# ============================================================
# Family B: LISAH n_floor refinement (OPTMix stays at n=2)
# ============================================================
cat > "${PS_DIR}/B_LISAH_n1.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 1}}
EOF
cat > "${PS_DIR}/B_LISAH_n3.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 3}}
EOF
cat > "${PS_DIR}/B_LISAH_n4.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 4}}
EOF
run_v "B_LISAH_n1" "${PS_DIR}/B_LISAH_n1.json"
run_v "B_LISAH_n3" "${PS_DIR}/B_LISAH_n3.json"
run_v "B_LISAH_n4" "${PS_DIR}/B_LISAH_n4.json"

# ============================================================
# Family C: LISAH ratio_floor stack on top of n=2
# ============================================================
cat > "${PS_DIR}/C_LISAH_r0.6_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.6, "n_floor": 2}}
EOF
cat > "${PS_DIR}/C_LISAH_r0.7_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.7, "n_floor": 2}}
EOF
cat > "${PS_DIR}/C_LISAH_r0.8_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"ratio_floor": 0.8, "n_floor": 2}}
EOF
run_v "C_LISAH_r0.6_n2" "${PS_DIR}/C_LISAH_r0.6_n2.json"
run_v "C_LISAH_r0.7_n2" "${PS_DIR}/C_LISAH_r0.7_n2.json"
run_v "C_LISAH_r0.8_n2" "${PS_DIR}/C_LISAH_r0.8_n2.json"

# ============================================================
# Family D: OPTMix ratio_floor stack on top of n=2
# (D_r0.3 was hurting at β=0.10/n=30 — re-test under stronger demotion)
# ============================================================
cat > "${PS_DIR}/D_OPTMix_r0.3_n2.json" <<'EOF'
{"OPTMix": {"ratio_floor": 0.3, "n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/D_OPTMix_r0.4_n2.json" <<'EOF'
{"OPTMix": {"ratio_floor": 0.4, "n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/D_OPTMix_r0.6_n2.json" <<'EOF'
{"OPTMix": {"ratio_floor": 0.6, "n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
run_v "D_OPTMix_r0.3_n2" "${PS_DIR}/D_OPTMix_r0.3_n2.json"
run_v "D_OPTMix_r0.4_n2" "${PS_DIR}/D_OPTMix_r0.4_n2.json"
run_v "D_OPTMix_r0.6_n2" "${PS_DIR}/D_OPTMix_r0.6_n2.json"

# ============================================================
# Family E: RNNB / GUARDEN re-test at the new anchor
# ============================================================
cat > "${PS_DIR}/E_RNNB_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/E_GUARDEN_n2.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "GUARDEN": {"n_floor": 2}}
EOF
run_v "E_RNNB_n2"    "${PS_DIR}/E_RNNB_n2.json"
run_v "E_GUARDEN_n2" "${PS_DIR}/E_GUARDEN_n2.json"

# ============================================================
# Family F: anchor reproduction (sanity check)
# ============================================================
cat > "${PS_DIR}/F_anchor.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
run_v "F_anchor" "${PS_DIR}/F_anchor.json"

echo ""
echo "=== Stage 11 done. 15 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~7-9 LB):"
echo ""
echo "Round 1 (4 LB) — most likely shifts at β=0.15:"
echo "  D_OPTMix_r0.3_n2 (was hurt at β=0.10 — geo demotion may have flipped sign)"
echo "  C_LISAH_r0.6_n2 (Stage 3 marginal +0.00054 at β=0.10 — re-test at new lock)"
echo "  A_OPTMix_n3 (confirms OPTMix n=2 is still the local minimum)"
echo "  B_LISAH_n3 (confirms LISAH n=2 is still the local minimum)"
echo ""
echo "Round 2 (3-4 LB) — based on Round 1 winners:"
echo "  If D_OPTMix_r0.3_n2 wins: try D_r0.4_n2 to bracket"
echo "  If C_LISAH_r0.6_n2 wins: try C_r0.7_n2 to bracket"
echo "  Else: F_anchor sanity check + E_RNNB_n2 / E_GUARDEN_n2"
echo ""
echo "Stretch ceiling: 0.510-0.515 if any per-source shift compounds."
echo "Conservative outcome: 0.50488 stays the lock."
