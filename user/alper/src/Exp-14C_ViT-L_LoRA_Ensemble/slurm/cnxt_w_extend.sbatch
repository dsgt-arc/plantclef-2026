#!/bin/bash
# Exp-14C — extend w sweep above 0.90 to find the actual peak.
#
# cnxt_optimization_sweep results (beta=0.70, thr=0.085):
#   w=0.65: 0.43153 / 0.70: 0.43401 / 0.75: 0.43320 / 0.80: 0.43730
#   w=0.85: 0.44019 / 0.90: 0.44830 ← still climbing
#
# Trend monotonic increasing through w=0.90; this sweep tests w ∈ {0.92, 0.95, 0.97}.
# Also tests thr=0.080 alongside thr=0.085 in case threshold optimum
# also shifts at higher w.
#
# Usage:
#   sbatch slurm/cnxt_w_extend.sbatch

#SBATCH --job-name=exp14c-cnxt-wext
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
EXP14_SCRIPTS="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion/scripts"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

cd "${EXP14_ROOT}"

PREFUSION_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_cnxt_opt"

mkdir -p "${OUT_DIR}"

PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b0.70.parquet"
if [ ! -f "${PREFUSION}" ]; then
    echo "ERROR: missing prefusion parquet at ${PREFUSION}"
    exit 1
fi

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: extend w sweep above 0.90 (3 weights × 2 thresholds = 6 CSVs) ==="
echo "Reference: w=0.90, thr=0.085 = 0.44830 (current best, still climbing)"
echo "Output: ${OUT_DIR}"

python "${EXP14_SCRIPTS}/ensemble_v2.py" \
    --vit-prefusion-parquet "${PREFUSION}" \
    --convnext-scales \
        "${CNXT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_7x7.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --weights "0.92,0.95,0.97" \
    --thresholds "0.080,0.085" \
    --topk 10 \
    --vit-aggregator max \
    --cnxt-aggregator max \
    --tag "exp14c_cnxtopt_b0.70"

# Tag CSVs to distinguish from prior runs
for f in "${OUT_DIR}"/exp14c_cnxtopt_b0.70_*.csv; do
    [ -e "$f" ] || continue
    base="$(basename "$f" .csv)"
    case "$base" in
        *_vitLK_cnxtK) ;;
        *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtK.csv" ;;
    esac
done

rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true

echo ""
echo "=== Done. 6 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~3-5 LB):"
echo "  Round 1 (3 LB): w=0.92, 0.95, 0.97 all at thr=0.085 → find w peak"
echo "  Round 2 (1-2 LB): if peak at w=0.95+, also try thr=0.080 at peak"
echo ""
echo "Decision:"
echo "  - If w=0.95 or 0.97 > 0.45: meaningful new peak found"
echo "  - If w=0.92 wins or all decline from w=0.90: peak is around 0.90-0.92"
echo "  - Either way: lock the w*, then apply Exp-14 mechanisms (sim_w + masks)"
