#!/bin/bash
# Exp-16C Stage 7 — refine n_near_min around the new peak.
#
# Stage 5 Family C found n_near_min=40 → F1=0.50304 (BEATS 0.49896 target).
# Curve: n=20: 0.47722, n=30: 0.49207, n=40: 0.50304, n=50: 0.49997.
# Monotone up from 20→40, slight rollover at 50.
#
# This stage refines around the peak with n ∈ {25, 30, 35, 45}
# to confirm shape and re-anchor the locked config.
#
# Held: β=0.10, R=200, alt-on, thr=0.085, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 (Stage 4 lock)
#
# Total: 4 CSVs.
#
# Usage:
#   sbatch slurm/stage7_n_near_refine.sbatch

#SBATCH --job-name=exp16c-s7-nrefine
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage7"
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
echo "=== Exp-16C Stage 7: n_near refinement at new peak (4 CSVs) ==="
echo "Stage 5 best: n=40 → 0.50304 (TARGET 0.49896 CLEARED)"

run_v() {
    local TAG="$1"; local N_NEAR="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s7_${TAG}.csv"
    echo ""
    echo "--- ${TAG}: n_near=${N_NEAR} ---"
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
        --n-near-min "${N_NEAR}" \
        --use-altitude \
        --threshold 0.085 \
        --topk 10 \
        --n-floor 3 \
        --ratio-floor 0.5 \
        --per-source-json "${LOCKED_PS}" \
        --out-csv "${OUT_CSV}"
}

run_v "n25" 25
run_v "n30" 30
run_v "n35" 35
run_v "n45" 45

echo ""
echo "=== Stage 7 done. 4 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy: submit all 4 to characterize both flanks of the peak."
echo "Stage 5 anchors: n=20: 0.47722, n=30: 0.49207, n=40: 0.50304, n=50: 0.49997."
echo "n=45 brackets the n=40→50 rollover; n=25/30/35 fill the lower flank."
