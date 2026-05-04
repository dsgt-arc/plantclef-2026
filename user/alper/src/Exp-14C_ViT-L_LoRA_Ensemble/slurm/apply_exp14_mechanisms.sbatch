#!/bin/bash
# Exp-14C — apply the three Exp-14 mechanisms that lifted Exp-11B (0.44428)
# to Exp-14 raw best (0.45611):
#
#   1. Stage 4: per-source sim_w aggregator on RNNB + RNNB α=0.30 (+0.00982)
#   2. Stage 12: aquatic-species habitat mask (+0.00162)
#   3. Stage 13: alpine biogeo mask (+0.00039)
#
# Total expected lift over the no-mechanism ensemble: ~+0.012 F1.
#
# Procedure:
#   A) Run ensemble_v2.py with --vit-aggregator-per-source '{"RNNB":"sim_w"}',
#      --vit-controller vit_rnnb030.json, --save-ensemble-parquet
#      → produces both a CSV (sim_w only) and an ensemble parquet
#   B) apply_habitat_mask.py on the ensemble parquet → CSV with sim_w + habitat
#   C) apply_biogeo_mask.py on the same parquet → CSV with sim_w + habitat + biogeo
#
# Configurable via env vars (defaults are the fully-locked Exp-14C optimum):
#   W=0.88    (locked from cnxtP_w_sweep — peak shifted down 0.02 from K's 0.90)
#   THR=0.085 (locked across standalone, kNN-only, and ensemble sweeps)
#   BETA=0.70 (locked from kNN-only sweep)
#   ConvNeXt scheme: P (5+6+7) — locked from cnxt_scheme_sweep
#       Locked baseline F1 (no Exp-14 mechanisms): 0.44969
#
# Usage:
#   sbatch slurm/apply_exp14_mechanisms.sbatch
#   # or override the (w, thr) once cnxt_w_extend completes:
#   sbatch --export=W=0.95,THR=0.085 slurm/apply_exp14_mechanisms.sbatch

#SBATCH --job-name=exp14c-mechanisms
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

W="${W:-0.88}"
THR="${THR:-0.085}"
BETA="${BETA:-0.70}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
EXP14_SCRIPTS="${EXP14_ROOT}/scripts"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

cd "${EXP14_ROOT}"

PREFUSION_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
VIT_CONTROLLER="${EXP14_ROOT}/stage_configs/controllers/vit_rnnb030.json"
SIM_JSON="${EXP14_ROOT}/outputs/signals/visit_similarity.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms"

mkdir -p "${OUT_DIR}"

# Sanity checks on inputs
PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b${BETA}.parquet"
for f in "${PREFUSION}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${VIT_CONTROLLER}" "${SIM_JSON}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required file: $f"
        exit 1
    fi
done

TAG="exp14c_simw_b${BETA}_w${W}_thr${THR}"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: apply Exp-14 mechanisms (sim_w + habitat + biogeo) ==="
echo "Config: beta=${BETA}, w=${W}, thr=${THR}"
echo "Tag:    ${TAG}"
echo "Output: ${OUT_DIR}"

# ---------------------------------------------------------------
# Step A: ensemble_v2.py with sim_w on RNNB + α=0.30 controller
#         + save ensemble parquet for downstream mask application
# ---------------------------------------------------------------
echo ""
echo "--- Step A: ensemble with sim_w on RNNB ---"
python "${EXP14_SCRIPTS}/ensemble_v2.py" \
    --vit-prefusion-parquet "${PREFUSION}" \
    --convnext-scales \
        "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --weights "${W}" \
    --thresholds "${THR}" \
    --topk 10 \
    --vit-controller "${VIT_CONTROLLER}" \
    --vit-aggregator max \
    --vit-aggregator-per-source '{"RNNB":"sim_w"}' \
    --cnxt-aggregator max \
    --sim-json "${SIM_JSON}" \
    --save-ensemble-parquet \
    --tag "${TAG}"

# ensemble_v2 emits one CSV (sim_w only) plus an ensemble parquet
SIMW_CSV="${OUT_DIR}/${TAG}_w${W}_thr${THR}_top10.csv"
ENS_PARQUET="${OUT_DIR}/${TAG}_w${W}_ensemble.parquet"

# Tag the sim_w-only CSV
mv "${SIMW_CSV}" "${OUT_DIR}/${TAG}_simwOnly.csv" 2>/dev/null || true
SIMW_CSV="${OUT_DIR}/${TAG}_simwOnly.csv"

if [ ! -f "${ENS_PARQUET}" ]; then
    echo "ERROR: ensemble parquet not saved at ${ENS_PARQUET}"
    exit 1
fi

# ---------------------------------------------------------------
# Step B: apply habitat mask (Stage 12) — adds aquatic blacklist
# ---------------------------------------------------------------
echo ""
echo "--- Step B: apply aquatic habitat mask ---"
HABITAT_CSV="${OUT_DIR}/${TAG}_simw_habitat.csv"
python "${EXP14_SCRIPTS}/apply_habitat_mask.py" \
    --ensemble-parquet "${ENS_PARQUET}" \
    --species-map "${SPECIES_MAP}" \
    --geo-mask "${GEO_MASK}" \
    --habitat-mask "${HABITAT_MASK}" \
    --out-csv "${HABITAT_CSV}" \
    --threshold "${THR}" \
    --topk 10

# ---------------------------------------------------------------
# Step C: apply biogeo mask (Stage 13) — geo + habitat + alpine biogeo
# ---------------------------------------------------------------
echo ""
echo "--- Step C: apply biogeo mask (also includes geo + habitat) ---"
BIOGEO_CSV="${OUT_DIR}/${TAG}_simw_habitat_biogeo.csv"
python "${EXP14_SCRIPTS}/apply_biogeo_mask.py" \
    --ensemble-parquet "${ENS_PARQUET}" \
    --species-map "${SPECIES_MAP}" \
    --geo-mask "${GEO_MASK}" \
    --habitat-mask "${HABITAT_MASK}" \
    --biogeo-npz "${BIOGEO_NPZ}" \
    --out-csv "${BIOGEO_CSV}" \
    --threshold "${THR}" \
    --topk 10

echo ""
echo "=== Done. 3 CSVs produced ==="
echo "  ${SIMW_CSV}"
echo "  ${HABITAT_CSV}"
echo "  ${BIOGEO_CSV}"
echo ""
echo "Submission strategy (3 LB):"
echo "  1. sim_w only           → tests sim_w transfer (+0.010 expected)"
echo "  2. sim_w + habitat      → adds aquatic blacklist (+0.0016 expected)"
echo "  3. sim_w + habitat + biogeo → full Exp-14 raw recipe (+0.0004 more)"
echo ""
echo "Reference points:"
echo "  Current best (CNXT P, w=0.88, no masks): 0.44969"
echo "  Exp-11B production:                       0.44428"
echo "  Exp-14 raw best (target to beat):         0.45611"
echo "  Predicted with all 3 mechanisms:          ~0.462 (+0.006 over Exp-14)"
