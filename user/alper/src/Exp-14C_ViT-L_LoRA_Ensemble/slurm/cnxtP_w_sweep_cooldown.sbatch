#!/bin/bash
# Exp-14C — re-tune w sweep at ConvNeXt-cooldown (phase2v3) scheme P.
#
# Production ConvNeXt (phase2) + ViT-L (phase2v3) at scheme P found w=0.88
# was the peak (F1=0.44969 at base ensemble, before sim_w/habitat/biogeo).
# Cooldown ConvNeXt should be slightly stronger → optimal w may shift down
# (more ConvNeXt, less ViT-L).
#
# Sweep: w ∈ {0.80, 0.84, 0.86, 0.88, 0.90, 0.92} × thr=0.085 → 6 CSVs.
#
# Usage:
#   sbatch slurm/cnxtP_w_sweep_cooldown.sbatch

#SBATCH --job-name=exp14c-cnxtP-cooldown
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --qos=embers
#SBATCH --partition=cpu-medium
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
EXP14_SCRIPTS="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion/scripts"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
cd "${EXP14_ROOT}"

PREFUSION_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"
# COOLDOWN ConvNeXt parquets (NEW — produced by Exp-10C extract_phase2v3.sbatch)
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large_phase2v3"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_cnxt_cooldown_w"

mkdir -p "${OUT_DIR}"

PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b0.70.parquet"
for f in "${PREFUSION}" \
         "${CNXT_PARQUET_ROOT}/scale_5x5.parquet" \
         "${CNXT_PARQUET_ROOT}/scale_6x6.parquet" \
         "${CNXT_PARQUET_ROOT}/scale_7x7.parquet"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f"; exit 1; fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C: w sweep at ConvNeXt phase2v3 (cooldown) scheme P ==="
echo "Reference (production ConvNeXt): w=0.88 → 0.44969 base ensemble"
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
    --weights "0.80,0.84,0.86,0.88,0.90,0.92" \
    --thresholds "0.085" \
    --topk 10 \
    --vit-aggregator max \
    --cnxt-aggregator max \
    --tag "exp14c_b0.70_cnxtPcooldown"

for f in "${OUT_DIR}"/exp14c_b0.70_cnxtPcooldown_*.csv; do
    [ -e "$f" ] || continue
    base="$(basename "$f" .csv)"
    case "$base" in
        *_vitLK_cnxtPcooldown) ;;
        *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtPcooldown.csv" ;;
    esac
done
rm -f "${OUT_DIR}/_branch_cnxt_prefusion.parquet" || true

echo ""
echo "=== Done. 6 CSVs in ${OUT_DIR} ==="
echo ""
echo "Local sanity check (run on laptop with downloaded CSVs):"
echo "  python C:/Users/alper/Desktop/analyze_exp16c.py <csv>"
echo ""
echo "Submit 1 LB at most likely peak (start w=0.88, then bracket if shifted)."
echo "Lock the new w*, then submit apply_exp14_mechanisms_cooldown.sbatch with --export=W=w*"
