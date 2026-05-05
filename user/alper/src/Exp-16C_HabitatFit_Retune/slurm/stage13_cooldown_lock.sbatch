#!/bin/bash
# Exp-16C Stage 13 — re-run the locked Exp-16C config with the cooldown
# ConvNeXt ensemble parquet.
#
# Held: alt-on, β_global=0.15, R=200, n_near_min=40, thr=0.085, k=10
#       Per-source admission: OPTMix n=2 + LISAH n=2 + RNNB n=2
# This is the locked Exp-16C config (Stage 11 anchor F1=0.50507).
#
# The only thing changing is the source ensemble parquet (cooldown
# ConvNeXt instead of production ConvNeXt). Everything downstream
# (habitat-fit demotion + per-source admission) is identical.
#
# Override the ensemble parquet:
#   sbatch --export=ENS_PARQUET=/path/to/cooldown_ensemble.parquet \
#          slurm/stage13_cooldown_lock.sbatch
#
# Optional: also re-test with per-source β at OPTMix=0.05 if Stage 12
# A_OPTMix_b0.05 won. Set TEST_PERSRC_BETA=1 to enable.

#SBATCH --job-name=exp16c-s13-cooldown
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
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

# ENS_PARQUET MUST be passed via --export, e.g.
#   --export=ENS_PARQUET=$PLANTCLEF_ROOT/aerten3/submissions/exp14c_mechanisms_cooldown/exp14c_cooldown_simw_b0.70_w0.88_thr0.085_w0.88_ensemble.parquet
if [ -z "${ENS_PARQUET:-}" ]; then
    echo "ERROR: ENS_PARQUET not set. Pass via --export=ENS_PARQUET=/path/to/...ensemble.parquet"
    exit 1
fi

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage13_cooldown"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done

cat > "${PS_DIR}/locked.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
LOCKED_PS="${PS_DIR}/locked.json"

cat > "${PS_DIR}/locked_OPTMix_b0.05.json" <<'EOF'
{"OPTMix": {"n_floor": 2, "beta": 0.05}, "LISAH": {"n_floor": 2}, "RNNB": {"n_floor": 2}}
EOF
PS_BETA="${PS_DIR}/locked_OPTMix_b0.05.json"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 13: cooldown ConvNeXt at locked config ==="
echo "Ensemble parquet: ${ENS_PARQUET}"
echo "Lock: alt-on, β=0.15, R=200, n=40, thr=0.085, k=10"
echo "Admission: OPTMix n=2 + LISAH n=2 + RNNB n=2"
echo "Production-ConvNeXt anchor F1: 0.50507"

run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s13_${TAG}.csv"
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

run_v "locked"           "${LOCKED_PS}"
run_v "locked_OPTMix_b005" "${PS_BETA}"

echo ""
echo "=== Stage 13 done. 2 CSVs in ${OUT_DIR} ==="
echo ""
echo "Compare:"
echo "  exp16c_s13_locked.csv         (cooldown vs 0.50507 production ConvNeXt anchor)"
echo "  exp16c_s13_locked_OPTMix_b005.csv (cooldown + Stage 12 per-source β idea)"
echo ""
echo "Submit only one per LB day. Pick the more-promising one based on Stage 12 result."
