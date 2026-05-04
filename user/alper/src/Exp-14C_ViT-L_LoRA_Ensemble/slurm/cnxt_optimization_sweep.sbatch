#!/bin/bash
# Exp-14C — ConvNeXt-side optimization sweep at locked ViT-L+kNN optimum.
#
# kNN-only result: ViT-L+kNN at beta=0.70, thr=0.085 = 0.43688
# Stage 1 best (with ConvNeXt at w=0.80): 0.44253 (+0.006 lift only)
# This sweep tests whether pushing ConvNeXt to a HIGHER weight reveals more
# value, since the +0.006 marginal lift at w=0.80 suggests ConvNeXt is
# under-weighted (or the wrong w altogether).
#
# Sweep at fixed ViT-L+kNN config (beta=0.70 prefusion):
#   weight w   ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
#   threshold  ∈ {0.085, 0.095, 0.100}
# Total: 9 × 3 = 27 CSVs at scheme K (4+5+6+7) for ConvNeXt.
#
# Usage:
#   sbatch slurm/cnxt_optimization_sweep.sbatch

#SBATCH --job-name=exp14c-cnxt-opt
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
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

# Use the locked ViT-L+kNN winner: beta=0.70 prefusion
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
echo "=== Exp-14C: ConvNeXt-side optimization (9 weights × 3 thresholds = 27 CSVs) ==="
echo "Fixed: ViT-L+kNN beta=0.70 (prefusion locked), CNXT scheme K (4+5+6+7)"
echo "Output: ${OUT_DIR}"

# Single ensemble_v2.py call sweeps all weights and thresholds at once
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
    --weights "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90" \
    --thresholds "0.085,0.095,0.100" \
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

# Cleanup intermediate parquet
rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true

echo ""
echo "=== Done. 27 CSVs in ${OUT_DIR} ==="
echo ""
echo "Reference points:"
echo "  ViT-L + kNN only (no ConvNeXt) at thr=0.085:  0.43688"
echo "  Stage 1 best (w=0.80, thr=0.090):            0.44253"
echo "  Exp-11B production (ViT-B+kNN+CNXT, w=0.77): 0.44428"
echo "  Exp-14 raw (sim_w + biogeo):                 0.45611  ← target"
echo ""
echo "Submission strategy (~9 LB):"
echo "  Round 1 (3 LB): w=0.65, 0.75, 0.85 at thr=0.085 (map the w response curve)"
echo "  Round 2 (3 LB): at winning w, sweep thr=0.085, 0.095, 0.100"
echo "  Round 3 (3 LB): at winning (w, thr), narrow w by ±0.025"
echo ""
echo "What to look for:"
echo "  - If peak shifts BELOW w=0.77 (ConvNeXt up): old recipe was wrong, ConvNeXt is helpful"
echo "  - If peak stays around w=0.80: ConvNeXt's contribution is genuinely small"
echo "  - If peak ABOVE w=0.85 (ConvNeXt down): ConvNeXt is hurting at meaningful weights"
