#!/bin/bash
# Exp-16C Stage 6 — fix alpine over-saturation via per-source threshold.
#
# DIAGNOSTIC FINDING: 1,444/2,105 quadrats (CBN-PdlC + CBN-Pla) are at
# topk=10 cap (100% saturation) at the global thr=0.085. Alpine sources
# are dragging the F1 down by trading recall for catastrophic precision loss.
#
# Old pipeline at thr=0.100 didn't have this problem (alpine wasn't saturated).
# Our new ViT-L+kNN is much more confident on alpine, so we need to raise
# alpine threshold while keeping non-alpine sources at thr=0.085.
#
# Tests: per-source threshold for CBN-PdlC and CBN-Pla independently, then
# combined.
#
# Held: β=0.10, R=200, n_near=30, alt-on, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 (Stage 4 lock)
#       Global threshold: 0.085 (applied to non-alpine sources)
#
# Total: 16 CSVs.
#
# Note: apply_habitat_fit.py's --per-source-json supports a "threshold" key
# in addition to ratio_floor and n_floor. Verify by inspecting the script
# before running; if not supported, the script needs a small extension.

#SBATCH --job-name=exp16c-s6-alpine
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage6_alpine"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing required file: $f"; exit 1; fi
done

# IMPORTANT: This stage assumes apply_habitat_fit.py supports per-source threshold via "threshold" key
# in --per-source-json. If not, the script will fall back to global threshold for these sources
# and we'll see no effect. In that case, we need to extend apply_habitat_fit.py.

echo "=== Exp-16C Stage 6: alpine per-source threshold (16 CSVs) ==="
echo "Locked: β=0.10, R=200, n_near=30, alt-on, OPTMix n=2 + LISAH n=2"
echo "Stage 4 anchor: 0.49207"
echo "Diagnostic: alpine sources at 10-cap on 100% of quadrats; this is the bottleneck"

run_v() {
    local TAG="$1"; local PS_JSON="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s6_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: $(cat ${PS_JSON}) ---"
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
        --per-source-json "${PS_JSON}" \
        --out-csv "${OUT_CSV}"
}

# CBN-PdlC alone — 4 thresholds
for THR in 0.10 0.12 0.15 0.20; do
    CFG="${PS_DIR}/PdlC_t${THR}.json"
    cat > "${CFG}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "CBN-PdlC": {"threshold": ${THR}}}
EOF
    run_v "PdlC_t${THR}" "${CFG}"
done

# CBN-Pla alone — 4 thresholds
for THR in 0.10 0.12 0.15 0.20; do
    CFG="${PS_DIR}/Pla_t${THR}.json"
    cat > "${CFG}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "CBN-Pla": {"threshold": ${THR}}}
EOF
    run_v "Pla_t${THR}" "${CFG}"
done

# Both alpine sources combined — 4 same-threshold combos
for THR in 0.10 0.12 0.15 0.20; do
    CFG="${PS_DIR}/Both_t${THR}.json"
    cat > "${CFG}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "CBN-PdlC": {"threshold": ${THR}}, "CBN-Pla": {"threshold": ${THR}}}
EOF
    run_v "Both_t${THR}" "${CFG}"
done

# Asymmetric: CBN-PdlC tight, CBN-Pla loose (and vice versa) — 4 combos
for PDLC in 0.12 0.15; do
    for PLA in 0.10 0.12; do
        CFG="${PS_DIR}/Asym_PdlC${PDLC}_Pla${PLA}.json"
        cat > "${CFG}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "CBN-PdlC": {"threshold": ${PDLC}}, "CBN-Pla": {"threshold": ${PLA}}}
EOF
        run_v "Asym_PdlC${PDLC}_Pla${PLA}" "${CFG}"
    done
done

echo ""
echo "=== Stage 6 done. 16 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~6-8 LB):"
echo ""
echo "Round 1 (4 LB) — both alpine combined sweep:"
echo "  Both_t0.10, Both_t0.12, Both_t0.15, Both_t0.20"
echo "  → identifies the right alpine threshold floor"
echo ""
echo "Round 2 (3-4 LB) — at winning threshold, test single-source vs both:"
echo "  PdlC_t<winner>, Pla_t<winner>, Asym variants if asymmetric pattern emerges"
echo ""
echo "PRECONDITION CHECK:"
echo "  Run analyze script on PdlC_t0.15 BEFORE submitting — verify CBN-PdlC mean count"
echo "  drops below 10. If alpine still at 10-cap, apply_habitat_fit.py doesn't support"
echo "  per-source threshold and we need to extend the script."
echo ""
echo "Predicted gain if alpine threshold works: +0.005 to +0.015 → 0.497-0.507"
echo "Target: 0.49896. Should clear with margin."
