#!/bin/bash
# Exp-16C Stage 12 — per-source β at the locked anchor.
#
# Mechanism (NEW): apply_habitat_fit.py now supports a "beta" key in
# --per-source-json. When present, the fit table for that source is built
# with the override β instead of the global β=0.15.
#
# Hypothesis from per-source diagnosis:
#   - OPTMix has OOP species (Plantago subulata, Bothriochloa) → wants TIGHTER β
#   - Alpine sources (CBN-PdlC, CBN-Pla) at 100% topk-10 cap → MAY want
#     tighter β to push non-alpine intruders harder out of top-10,
#     OR looser β to lift recall on saturated sources.
#   - LISAH/GUARDEN/RNNB look biogeographically clean → likely no effect
#
# Held: alt-on, β_global=0.15, R=200, n_near_min=40, thr=0.085, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 + RNNB n=2 (Stage 11 lock)
#
# Sweeps:
#   A) OPTMix tighter   ∈ {0.05, 0.10}                          (2 variants)
#   B) Alpine tighter   ∈ {0.05, 0.10}    (PdlC+Pla together)   (2 variants)
#   C) Alpine looser    ∈ {0.30, 0.50, 1.0}                     (3 variants)
#   D) Combined         OPTMix b0.05 + alpine b0.10,
#                       OPTMix b0.10 + alpine b0.50              (2 variants)
#   F) Anchor reproduction                                       (1 variant)
#
# Total: 10 CSVs.
#
# Usage:
#   sbatch slurm/stage12_per_source_beta.sbatch

#SBATCH --job-name=exp16c-s12-perbeta
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage12"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing required file: $f"; exit 1; fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 12: per-source β at locked anchor (10 CSVs) ==="
echo "Lock: alt-on, β_global=0.15, R=200, n=40, thr=0.085, k=10"
echo "Admission: OPTMix n=2 + LISAH n=2 + RNNB n=2"
echo "Anchor F1: 0.50507 (Stage 11 E_RNNB_n2)"

run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s12_${TAG}.csv"
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
# Family A: OPTMix tighter β (target Plantago-subulata-style OOP species)
# ============================================================
cat > "${PS_DIR}/A_OPTMix_b0.05.json" <<'EOF'
{"OPTMix": {"n_floor": 2, "beta": 0.05}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
cat > "${PS_DIR}/A_OPTMix_b0.10.json" <<'EOF'
{"OPTMix": {"n_floor": 2, "beta": 0.10}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
run_v "A_OPTMix_b0.05" "${PS_DIR}/A_OPTMix_b0.05.json"
run_v "A_OPTMix_b0.10" "${PS_DIR}/A_OPTMix_b0.10.json"

# ============================================================
# Family B: Alpine tighter β (push non-alpine intruders harder out of top-10)
# ============================================================
cat > "${PS_DIR}/B_alpine_b0.05.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.05}, "CBN-Pla": {"beta": 0.05}}
EOF
cat > "${PS_DIR}/B_alpine_b0.10.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.10}, "CBN-Pla": {"beta": 0.10}}
EOF
run_v "B_alpine_b0.05" "${PS_DIR}/B_alpine_b0.05.json"
run_v "B_alpine_b0.10" "${PS_DIR}/B_alpine_b0.10.json"

# ============================================================
# Family C: Alpine looser β (test recall ceiling — alpine at 100% topk cap)
# ============================================================
cat > "${PS_DIR}/C_alpine_b0.30.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.30}, "CBN-Pla": {"beta": 0.30}}
EOF
cat > "${PS_DIR}/C_alpine_b0.50.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.50}, "CBN-Pla": {"beta": 0.50}}
EOF
cat > "${PS_DIR}/C_alpine_b1.00.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 1.00}, "CBN-Pla": {"beta": 1.00}}
EOF
run_v "C_alpine_b0.30" "${PS_DIR}/C_alpine_b0.30.json"
run_v "C_alpine_b0.50" "${PS_DIR}/C_alpine_b0.50.json"
run_v "C_alpine_b1.00" "${PS_DIR}/C_alpine_b1.00.json"

# ============================================================
# Family D: Combined (best-guess intersections)
# ============================================================
cat > "${PS_DIR}/D_OPTMix_b0.05_alpine_b0.10.json" <<'EOF'
{"OPTMix": {"n_floor": 2, "beta": 0.05}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.10}, "CBN-Pla": {"beta": 0.10}}
EOF
cat > "${PS_DIR}/D_OPTMix_b0.10_alpine_b0.50.json" <<'EOF'
{"OPTMix": {"n_floor": 2, "beta": 0.10}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.50}, "CBN-Pla": {"beta": 0.50}}
EOF
run_v "D_OPTMix_b0.05_alpine_b0.10" "${PS_DIR}/D_OPTMix_b0.05_alpine_b0.10.json"
run_v "D_OPTMix_b0.10_alpine_b0.50" "${PS_DIR}/D_OPTMix_b0.10_alpine_b0.50.json"

# ============================================================
# Family F: anchor reproduction (sanity)
# ============================================================
cat > "${PS_DIR}/F_anchor.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
run_v "F_anchor" "${PS_DIR}/F_anchor.json"

echo ""
echo "=== Stage 12 done. 10 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~5-7 LB):"
echo ""
echo "Round 1 (3 LB) — anchor + the two most likely directions:"
echo "  F_anchor (must reproduce 0.50507; validates the per-source β code path)"
echo "  A_OPTMix_b0.05 (most likely lift if OOP demotion fixes Plantago subulata)"
echo "  B_alpine_b0.10 (most likely lift if alpine intruders matter)"
echo ""
echo "Round 2 (2-3 LB) — based on Round 1:"
echo "  If A wins: try A_OPTMix_b0.10 to bracket"
echo "  If B wins: try B_alpine_b0.05 to push further"
echo "  If neither: try C_alpine_b0.50 and C_alpine_b1.00 (recall-ceiling test)"
echo ""
echo "Round 3 (1-2 LB) — combos:"
echo "  D variants if both A and B wins (for additivity test)"
echo ""
echo "Predicted ceiling: 0.508-0.513 if any per-source β shifts meaningfully."
echo "Conservative outcome: 0.50507 stays the lock."
