#!/bin/bash
# Exp-16C Stage 12B — refine alpine β around the new peak.
#
# Stage 12 found alpine β=0.50 → 0.50558 (best so far). Curve at fixed
# global β=0.15:
#   alpine β=0.05: 0.50516
#   alpine β=0.10: 0.50515
#   alpine β=0.15: 0.50507 (anchor; global β applied uniformly)
#   alpine β=0.30: 0.50551
#   alpine β=0.50: 0.50558 (peak)
#   alpine β=1.00: 0.49106 (catastrophic — non-alpine intruders dominate)
#
# Cliff between 0.50 and 1.00. Refine:
#   - 0.40, 0.45 → bracket below the peak
#   - 0.55, 0.60 → bracket above the peak
#   - 0.70 → mid-cliff to localize where the regression starts
#
# Held: alt-on, β_global=0.15, R=200, n_near_min=40, thr=0.085, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 + RNNB n=2
#
# Total: 6 CSVs (5 alpine β + 1 anchor sanity).
#
# Usage:
#   sbatch slurm/stage12B_alpine_beta_refine.sbatch

#SBATCH --job-name=exp16c-s12B-alpine
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

ENS_PARQUET="${ENS_PARQUET:-${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms/exp14c_simw_b0.70_w0.88_thr0.085_w0.88_ensemble.parquet}"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage12B"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 12B: refine alpine β around peak (6 CSVs) ==="
echo "Lock: alt-on, β_global=0.15, R=200, n=40, thr=0.085, k=10"
echo "Admission: OPTMix n=2 + LISAH n=2 + RNNB n=2"
echo "Stage 12 anchor (alpine β=0.50): 0.50558"

run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s12B_${TAG}.csv"
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

# Helper: write per-source JSON for symmetric alpine β
write_alpine_json() {
    local PATH_OUT="$1"; local BETA="$2"
    cat > "${PATH_OUT}" <<EOF
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2},
 "CBN-PdlC": {"beta": ${BETA}}, "CBN-Pla": {"beta": ${BETA}}}
EOF
}

# Symmetric alpine β refinement
write_alpine_json "${PS_DIR}/alpine_b0.40.json" 0.40
write_alpine_json "${PS_DIR}/alpine_b0.45.json" 0.45
write_alpine_json "${PS_DIR}/alpine_b0.55.json" 0.55
write_alpine_json "${PS_DIR}/alpine_b0.60.json" 0.60
write_alpine_json "${PS_DIR}/alpine_b0.70.json" 0.70

run_v "alpine_b0.40" "${PS_DIR}/alpine_b0.40.json"
run_v "alpine_b0.45" "${PS_DIR}/alpine_b0.45.json"
run_v "alpine_b0.55" "${PS_DIR}/alpine_b0.55.json"
run_v "alpine_b0.60" "${PS_DIR}/alpine_b0.60.json"
run_v "alpine_b0.70" "${PS_DIR}/alpine_b0.70.json"

# Anchor reproduction (alpine β=0.50, must reproduce 0.50558)
write_alpine_json "${PS_DIR}/anchor_alpine_b0.50.json" 0.50
run_v "anchor_alpine_b0.50" "${PS_DIR}/anchor_alpine_b0.50.json"

echo ""
echo "=== Stage 12B done. 6 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~3 LB):"
echo ""
echo "Round 1 (1 LB) — anchor sanity:"
echo "  anchor_alpine_b0.50 → must reproduce 0.50558"
echo ""
echo "Round 2 (2 LB) — symmetric peaks around 0.50:"
echo "  alpine_b0.45 (closest below)"
echo "  alpine_b0.55 (closest above)"
echo "  → if either beats 0.50558, the peak shifted; bracket further"
echo ""
echo "Round 3 (optional 1 LB) — based on Round 2 result:"
echo "  if 0.45 wins: try 0.40"
echo "  if 0.55 wins: try 0.60"
echo "  if 0.50 still wins: stop, lock it"
echo ""
echo "Predicted ceiling at this anchor: 0.506-0.508."
