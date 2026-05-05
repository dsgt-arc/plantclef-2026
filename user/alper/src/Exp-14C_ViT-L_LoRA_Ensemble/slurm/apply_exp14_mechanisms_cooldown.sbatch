#!/bin/bash
# Exp-14C — apply Exp-14 mechanisms with the COOLDOWN ConvNeXt branch.
#
# Same pipeline as apply_exp14_mechanisms.sbatch but using
# convnext_large_phase2v3 parquets instead of convnext_large.
#
# Output ensemble parquet feeds Stage 13+ Exp-16C pipeline:
#   exp14c_simw_b<BETA>_w<W>_thr<THR>_w<W>_ensemble.parquet
#
# Configurable via env vars:
#   W=0.88    (default; override after cnxtP_w_sweep_cooldown finishes)
#   THR=0.085
#   BETA=0.70
#
# Usage:
#   sbatch slurm/apply_exp14_mechanisms_cooldown.sbatch
#   sbatch --export=W=0.86,THR=0.085 slurm/apply_exp14_mechanisms_cooldown.sbatch

#SBATCH --job-name=exp14c-mechanisms-cooldown
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
# COOLDOWN ConvNeXt parquets
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large_phase2v3"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
VIT_CONTROLLER="${EXP14_ROOT}/stage_configs/controllers/vit_rnnb030.json"
SIM_JSON="${EXP14_ROOT}/outputs/signals/visit_similarity.json"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms_cooldown"
mkdir -p "${OUT_DIR}"

PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b${BETA}.parquet"
for f in "${PREFUSION}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" "${VIT_CONTROLLER}" "${SIM_JSON}" \
         "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
         "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
         "${CNXT_PARQUET_ROOT}/scale_7x7.parquet"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done

TAG="exp14c_cooldown_simw_b${BETA}_w${W}_thr${THR}"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: apply Exp-14 mechanisms with cooldown ConvNeXt ==="
echo "Config: beta=${BETA}, w=${W}, thr=${THR}"
echo "ConvNeXt: ${CNXT_PARQUET_ROOT}"
echo "Tag:    ${TAG}"
echo "Output: ${OUT_DIR}"

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

SIMW_CSV="${OUT_DIR}/${TAG}_w${W}_thr${THR}_top10.csv"
ENS_PARQUET="${OUT_DIR}/${TAG}_w${W}_ensemble.parquet"
mv "${SIMW_CSV}" "${OUT_DIR}/${TAG}_simwOnly.csv" 2>/dev/null || true
SIMW_CSV="${OUT_DIR}/${TAG}_simwOnly.csv"

if [ ! -f "${ENS_PARQUET}" ]; then
    echo "ERROR: ensemble parquet not saved at ${ENS_PARQUET}"; exit 1
fi

echo ""
echo "--- Step B: aquatic habitat mask ---"
HABITAT_CSV="${OUT_DIR}/${TAG}_simw_habitat.csv"
python "${EXP14_SCRIPTS}/apply_habitat_mask.py" \
    --ensemble-parquet "${ENS_PARQUET}" \
    --species-map "${SPECIES_MAP}" \
    --geo-mask "${GEO_MASK}" \
    --habitat-mask "${HABITAT_MASK}" \
    --out-csv "${HABITAT_CSV}" \
    --threshold "${THR}" \
    --topk 10

echo ""
echo "--- Step C: biogeo mask (geo + habitat + alpine) ---"
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
echo "=== Done. 3 CSVs + 1 ensemble parquet produced ==="
echo "  ${SIMW_CSV}"
echo "  ${HABITAT_CSV}"
echo "  ${BIOGEO_CSV}"
echo "  ENS_PARQUET=${ENS_PARQUET}"
echo ""
echo "Reference (production ConvNeXt): biogeo CSV scored 0.45728"
echo ""
echo "NEXT: feed ENS_PARQUET into Exp-16C Stage 13:"
echo "  sbatch --export=ENS_PARQUET=${ENS_PARQUET} \\"
echo "      ../../Exp-16C_HabitatFit_Retune/slurm/stage13_cooldown_lock.sbatch"
