#!/bin/bash
# Exp-16C Stage 14B — refine RNNB β around 0.50.
# RNNB β=0.50 = 0.50876 (best); RNNB β=0.05 = 0.50647. Anchor (no RNNB β override) = 0.50623.

#SBATCH --job-name=exp16c-s14B-rnnb
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage14B"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s14B_${TAG}.csv"
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

write_rnnb_json() {
    local PATH_OUT="$1"; local BETA="$2"
    cat > "${PATH_OUT}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2},
 "RNNB":  {"n_floor": 2, "beta": ${BETA}},
 "CBN-PdlC": {"beta": 0.45}, "CBN-Pla": {"beta": 0.45}}
EOF
}

for B in 0.30 0.40 0.60 0.70; do
    write_rnnb_json "${PS_DIR}/RNNB_b${B}.json" ${B}
    run_v "RNNB_b${B}" "${PS_DIR}/RNNB_b${B}.json"
done

# Anchor sanity (RNNB β=0.50, must reproduce 0.50876)
write_rnnb_json "${PS_DIR}/RNNB_b0.50_anchor.json" 0.50
run_v "RNNB_b0.50_anchor" "${PS_DIR}/RNNB_b0.50_anchor.json"

echo ""; echo "=== Stage 14B done. 5 CSVs in ${OUT_DIR} ==="
