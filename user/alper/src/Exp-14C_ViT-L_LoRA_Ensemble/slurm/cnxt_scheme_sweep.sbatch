#!/bin/bash
# Exp-14C — test alternative ConvNeXt tile schemes at locked ensemble config.
#
# All current Exp-14C runs used ConvNeXt scheme K (4+5+6+7, 126 tiles)
# because we matched ViT-L K. But:
#   - ConvNeXt's Exp-11B production was scheme E (4+5+6, 77 tiles)
#   - Scheme P (5+6+7) wasn't tested either
#
# This sweep holds ViT-L+kNN at the locked (beta=0.70, w=0.90, thr=0.085)
# config and varies ConvNeXt scheme:
#   E:  4+5+6      (77 tiles, ConvNeXt original optimum)
#   P:  5+6+7      (110 tiles, drops scale 4)
#   (K is already tested at 0.44830, no need to re-run)
#
# Each scheme gets thr ∈ {0.080, 0.085, 0.095} to catch any threshold shift.
# = 2 schemes × 3 thresholds = 6 CSVs.
#
# Usage:
#   sbatch slurm/cnxt_scheme_sweep.sbatch

#SBATCH --job-name=exp14c-cnxt-scheme
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
OUT_BASE="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_cnxt_scheme"

mkdir -p "${OUT_BASE}"

PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b0.70.parquet"
if [ ! -f "${PREFUSION}" ]; then
    echo "ERROR: missing prefusion parquet at ${PREFUSION}"
    exit 1
fi
for s in 4 5 6 7; do
    f="${CNXT_PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing ConvNeXt scale ${s}x${s} at $f"
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: ConvNeXt scheme variation at locked ensemble ==="
echo "Fixed: beta=0.70, w=0.90"
echo "Reference (CNXT scheme K, thr=0.085): 0.44830"

# ----------------------------------------
# Scheme E: 4+5+6
# ----------------------------------------
OUT_DIR="${OUT_BASE}/cnxtE"
mkdir -p "${OUT_DIR}"
echo ""
echo "--- ConvNeXt scheme E (4+5+6) ---"
python "${EXP14_SCRIPTS}/ensemble_v2.py" \
    --vit-prefusion-parquet "${PREFUSION}" \
    --convnext-scales \
        "${CNXT_PARQUET_ROOT}/scale_4x4.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
        "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --weights "0.90" \
    --thresholds "0.080,0.085,0.095" \
    --topk 10 \
    --vit-aggregator max \
    --cnxt-aggregator max \
    --tag "exp14c_b0.70_cnxtE"

for f in "${OUT_DIR}"/exp14c_b0.70_cnxtE_*.csv; do
    [ -e "$f" ] || continue
    base="$(basename "$f" .csv)"
    case "$base" in
        *_vitLK_cnxtE) ;;
        *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtE.csv" ;;
    esac
done
rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true

# ----------------------------------------
# Scheme P: 5+6+7
# ----------------------------------------
OUT_DIR="${OUT_BASE}/cnxtP"
mkdir -p "${OUT_DIR}"
echo ""
echo "--- ConvNeXt scheme P (5+6+7) ---"
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
    --weights "0.90" \
    --thresholds "0.080,0.085,0.095" \
    --topk 10 \
    --vit-aggregator max \
    --cnxt-aggregator max \
    --tag "exp14c_b0.70_cnxtP"

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
echo "=== Done. 6 CSVs total ==="
echo "  Scheme E (4+5+6):  ${OUT_BASE}/cnxtE/  → 3 CSVs at thr {0.080, 0.085, 0.095}"
echo "  Scheme P (5+6+7):  ${OUT_BASE}/cnxtP/  → 3 CSVs at thr {0.080, 0.085, 0.095}"
echo ""
echo "Submission strategy (~2 LB):"
echo "  Submit cnxtE_thr0.085 and cnxtP_thr0.085 → compare to cnxtK_thr0.085 (0.44830)"
echo ""
echo "Decision:"
echo "  - cnxtE > 0.450 OR cnxtP > 0.450  → switch ConvNeXt scheme, re-tune w"
echo "  - within ±0.003 of 0.44830        → keep K (no churn justified)"
echo "  - both worse                       → K is genuinely the right ConvNeXt scheme too"
