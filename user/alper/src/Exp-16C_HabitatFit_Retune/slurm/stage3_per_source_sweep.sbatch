#!/bin/bash
# Exp-16C Stage 3 — per-source admission re-tune from scratch.
#
# Held constant at new locked Stage 2B point:
#   β=0.10, R=200, n_near=30, alt-on, refined centroids, thr=0.085, k=10
# Baseline at this point (no per-source admission): 0.48830 (Stage 2A b0.10)
# Reference (with old B_OPTMix_n2 admission): 0.48992 (Stage 2B I_b0.10_admB)
#
# DESIGN: vary ONE source's admission while holding all others at defaults
# (ratio_floor=0.5, n_floor=3). Each variant isolates that source's contribution.
# Stage 4 will combine independent winners.
#
# For each of {RNNB, LISAH, OPTMix, GUARDEN}:
#   - ratio_floor sweep at default n=3:  r ∈ {0.3, 0.4, 0.6, 0.7, 0.8}  (5 variants)
#   - n_floor sweep at default r=0.5:    n ∈ {1, 2, 4, 5}              (4 variants)
# Per source: 9 variants. Total: 4 × 9 = 36 CSVs.
#
# Usage:
#   sbatch slurm/stage3_per_source_sweep.sbatch

#SBATCH --job-name=exp16c-s3-persource
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
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

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp16c_stage3"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

for f in "${ENS_PARQUET}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"; exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-16C Stage 3: per-source admission re-tune (36 CSVs) ==="
echo "Held: β=0.10, R=200, n_near=30, alt-on, thr=0.085, k=10"
echo "Default for unswept sources: ratio_floor=0.5, n_floor=3"

# Helper: write per-source JSON, then call apply_habitat_fit.py
run_v() {
    local TAG="$1"; local PS_JSON_PATH="$2"
    local OUT_CSV="${OUT_DIR}/exp16c_s3_${TAG}.csv"
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

# Generate config + run for each (source, axis, value) triple
for SRC in RNNB LISAH OPTMix GUARDEN; do
    # ratio_floor sweep (n_floor stays at default 3)
    for R in 0.3 0.4 0.6 0.7 0.8; do
        TAG="${SRC}_r${R}"
        CFG="${PS_DIR}/${TAG}.json"
        printf '{"%s": {"ratio_floor": %s}}\n' "${SRC}" "${R}" > "${CFG}"
        run_v "${TAG}" "${CFG}"
    done
    # n_floor sweep (ratio_floor stays at default 0.5)
    for N in 1 2 4 5; do
        TAG="${SRC}_n${N}"
        CFG="${PS_DIR}/${TAG}.json"
        printf '{"%s": {"n_floor": %s}}\n' "${SRC}" "${N}" > "${CFG}"
        run_v "${TAG}" "${CFG}"
    done
done

echo ""
echo "=== Stage 3 done. 36 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~16-20 LB across 2 days):"
echo ""
echo "Round 1 (8 LB) — most likely lifts (per-source ratio_floor sweep, top end):"
echo "  RNNB_r0.3, RNNB_r0.7"
echo "  LISAH_r0.6, LISAH_r0.7  (LISAH r=0.6 was old win; verify)"
echo "  OPTMix_r0.3, OPTMix_r0.4 (OPTMix r=0.3 was old win; verify)"
echo "  GUARDEN_r0.4, GUARDEN_r0.6"
echo ""
echo "Round 2 (8 LB) — n_floor variants for sources with promising r results:"
echo "  Sources where Round 1 found a win: test n_floor variations"
echo "  Plus OPTMix_n2 (old win) for direct comparison to ratio-only setup"
echo ""
echo "Decision:"
echo "  Lock independent winners → Stage 4 combines them and tests additivity"
echo "  Expected combined lift: +0.005 to +0.012 → projected 0.495-0.502"
