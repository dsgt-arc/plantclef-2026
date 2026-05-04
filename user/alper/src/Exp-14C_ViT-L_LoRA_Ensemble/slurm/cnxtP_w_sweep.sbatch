#!/bin/bash
# Exp-14C — refine w sweep for ConvNeXt scheme P (5+6+7).
#
# Scheme K w-peak was at w=0.90. Scheme P has 13% fewer ConvNeXt tiles
# (110 vs 126), so the effective ConvNeXt contribution per unit w is
# smaller — peak might shift to LOWER w (more ConvNeXt to compensate).
#
# Already tested at scheme P:
#   w=0.90, thr=0.085 = 0.44900 (best so far)
#
# Sweep: w ∈ {0.82, 0.85, 0.88, 0.92, 0.95} × thr=0.085 = 5 CSVs.
# Skips w=0.90 (have it).
#
# Usage:
#   sbatch slurm/cnxtP_w_sweep.sbatch

#SBATCH --job-name=exp14c-cnxtP-w
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --qos=embers
#SBATCH --partition=cpu-medium
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
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_cnxt_scheme/cnxtP"

mkdir -p "${OUT_DIR}"

PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b0.70.parquet"
if [ ! -f "${PREFUSION}" ]; then
    echo "ERROR: missing prefusion parquet at ${PREFUSION}"
    exit 1
fi

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: w sweep at ConvNeXt scheme P (5+6+7) ==="
echo "Reference: w=0.90, thr=0.085 (CNXT P) = 0.44900"
echo "Output:    ${OUT_DIR}"

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
    --weights "0.82,0.85,0.88,0.92,0.95" \
    --thresholds "0.085" \
    --topk 10 \
    --vit-aggregator max \
    --cnxt-aggregator max \
    --tag "exp14c_b0.70_cnxtP"

# Tag CSVs to distinguish from K-scheme runs
for f in "${OUT_DIR}"/exp14c_b0.70_cnxtP_*.csv; do
    [ -e "$f" ] || continue
    base="$(basename "$f" .csv)"
    case "$base" in
        *_vitLK_cnxtP) ;;
        *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtP.csv" ;;
    esac
done

rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true

echo ""
echo "=== Done. 5 CSVs in ${OUT_DIR} ==="
echo ""
echo "Submission strategy (~3-5 LB):"
echo "  Round 1 (3 LB): w=0.85, 0.88, 0.92 at thr=0.085 → bracket peak"
echo "  Round 2 (2 LB): w=0.82 and 0.95 if peak shifted to edge"
echo ""
echo "Decision:"
echo "  - peak stays at w=0.90: lock and proceed to apply_exp14_mechanisms"
echo "  - peak shifts to lower w (0.85-0.88): more ConvNeXt helps with P"
echo "  - peak shifts higher (0.92): scheme P prefers even less ConvNeXt"
echo "  - lock the new w*, then re-submit apply_exp14_mechanisms with --export=W=w*"
