#!/bin/bash
# Exp-16C Stage 14 — per-source β at locked alpine_b0.45 anchor (F1=0.50623).
# Tests untested sources (CBN-can, RNNB, GUARDEN, LISAH), retests OPTMix,
# and alpine asymmetric.

#SBATCH --job-name=exp16c-s14-perbeta2
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

ENS_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms/exp14c_simw_b0.70_w0.88_thr0.085_w0.88_ensemble.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage14"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

# Locked admission + alpine β=0.45 baseline. Each variant ADDS one per-source β override.
BASE='"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}, "CBN-PdlC": {"beta": 0.45}, "CBN-Pla": {"beta": 0.45}'

run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s14_${TAG}.csv"
    echo ""; echo "--- ${TAG} ---"
    python "${EXP16_SCRIPTS}/apply_habitat_fit.py" \
        --ensemble-parquet "${ENS_PARQUET}" --geo-profile "${GEO_PROFILE}" \
        --species-map "${SPECIES_MAP}" --geo-mask "${GEO_MASK}" \
        --habitat-mask "${HABITAT_MASK}" --biogeo-npz "${BIOGEO_NPZ}" \
        --refined-ref-json "${REFINED_REF}" --radius-km 200 --beta 0.15 \
        --n-near-min 40 --use-altitude --threshold 0.085 --topk 10 \
        --n-floor 3 --ratio-floor 0.5 --per-source-json "${PS_JSON_PATH}" \
        --out-csv "${OUT_CSV}"
}

# Helper: writes JSON with the base + an extra override for one source
add_override() {
    local NAME="$1"; local SRC="$2"; local KEY="$3"; local VAL="$4"
    local PATH_OUT="${PS_DIR}/${NAME}.json"
    # Merge: if SRC is alpine, override its existing beta; else add as new
    if [ "$SRC" = "CBN-PdlC" ] || [ "$SRC" = "CBN-Pla" ]; then
        # Replace alpine beta with override — rebuild from scratch
        cat > "${PATH_OUT}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": ${VAL}}, "CBN-Pla": {"beta": ${VAL}}}
EOF
    elif [ "$SRC" = "OPTMix" ] || [ "$SRC" = "LISAH" ] || [ "$SRC" = "RNNB" ]; then
        # Source already in base with n_floor — add beta to it
        cat > "${PATH_OUT}" <<EOF
{"OPTMix": {"n_floor": 2$([ "$SRC" = "OPTMix" ] && echo ", \"beta\": ${VAL}")},
 "LISAH": {"n_floor": 2$([ "$SRC" = "LISAH" ] && echo ", \"beta\": ${VAL}")},
 "RNNB":  {"n_floor": 2$([ "$SRC" = "RNNB" ] && echo ", \"beta\": ${VAL}")},
 "CBN-PdlC": {"beta": 0.45}, "CBN-Pla": {"beta": 0.45}}
EOF
    else
        # New source (CBN-can, GUARDEN) — just add beta
        cat > "${PATH_OUT}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.45}, "CBN-Pla": {"beta": 0.45},
 "${SRC}": {"beta": ${VAL}}}
EOF
    fi
    run_v "${NAME}" "${PATH_OUT}"
}

# OPTMix retest
add_override "OPTMix_b0.05" OPTMix beta 0.05
add_override "OPTMix_b0.30" OPTMix beta 0.30
add_override "OPTMix_b0.50" OPTMix beta 0.50

# CBN-can
add_override "CBNcan_b0.05" CBN-can beta 0.05
add_override "CBNcan_b0.30" CBN-can beta 0.30
add_override "CBNcan_b0.50" CBN-can beta 0.50

# RNNB
add_override "RNNB_b0.05" RNNB beta 0.05
add_override "RNNB_b0.50" RNNB beta 0.50

# GUARDEN
add_override "GUARDEN_b0.30" GUARDEN beta 0.30
add_override "GUARDEN_b0.50" GUARDEN beta 0.50

# LISAH
add_override "LISAH_b0.30" LISAH beta 0.30
add_override "LISAH_b0.50" LISAH beta 0.50

# Alpine asymmetric
cat > "${PS_DIR}/alpine_asym_PdlC0.45_Pla0.30.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.45}, "CBN-Pla": {"beta": 0.30}}
EOF
run_v "alpine_asym_PdlC0.45_Pla0.30" "${PS_DIR}/alpine_asym_PdlC0.45_Pla0.30.json"

cat > "${PS_DIR}/alpine_asym_PdlC0.30_Pla0.45.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.30}, "CBN-Pla": {"beta": 0.45}}
EOF
run_v "alpine_asym_PdlC0.30_Pla0.45" "${PS_DIR}/alpine_asym_PdlC0.30_Pla0.45.json"

# Anchor sanity (must reproduce 0.50623)
cat > "${PS_DIR}/anchor.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": 0.45}, "CBN-Pla": {"beta": 0.45}}
EOF
run_v "anchor" "${PS_DIR}/anchor.json"

echo ""; echo "=== Stage 14 done. 15 CSVs in ${OUT_DIR} ==="
