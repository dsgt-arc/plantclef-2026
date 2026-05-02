#!/bin/bash
# Exp-14C Stage 1b: blend the new ViT-L pre-fusion parquets with ConvNeXt K
# via Exp-14 ensemble_v2.py. Sweeps weight w and threshold for each beta.
#
# Calls ensemble_v2.py with baseline ControllerConfig (no per-source aggregators,
# no Exp-16 mechanisms) for the cleanest "is the new branch better?" signal.
#
# Stage-1 sweep (per beta_multi parquet from build_prefusion_sweep.sbatch):
#   weight w        ∈ {0.75, 0.80, 0.85}     → 3 values
#   threshold       ∈ {0.075, 0.090}         → 2 values
# 3 betas × 3 weights × 2 thresholds = 18 CSVs total.
#
# Usage:
#   sbatch slurm/ensemble_sweep_stage1.sbatch

#SBATCH --job-name=exp14c-ens-stage1
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
SRC_ROOT="${REPO_ROOT}/user/alper/src/Exp-14C_ViT-L_LoRA_Ensemble"
EXP14_SCRIPTS="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion/scripts"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"

mkdir -p logs

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

# ensemble_v2.py imports rely on Exp-14 sys.path setup (controller, signals, etc.)
# Set CWD to Exp-14 root so its relative sys.path inserts work.
cd "${EXP14_ROOT}"

PREFUSION_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"
CNXT_PARQUET_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
OUT_BASE="${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_stage1"

mkdir -p "${OUT_BASE}"

# Sanity checks
for b in 0.75 0.85 0.95; do
    f="${PREFUSION_DIR}/vitL_prefusion_b${b}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing prefusion parquet at $f"
        echo "Run build_prefusion_sweep.sbatch first."
        exit 1
    fi
done
for s in 4 5 6 7; do
    f="${CNXT_PARQUET_ROOT}/scale_${s}x${s}.parquet"
    if [ ! -f "$f" ]; then
        echo "ERROR: missing ConvNeXt scale ${s}x${s} at $f"
        exit 1
    fi
done

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Exp-14C Stage 1b: ensemble sweep (3 betas × 3 weights × 2 thresholds = 18 CSVs) ==="

# Loop over the 3 prefusion parquets (one per beta_multi)
for BETA in 0.75 0.85 0.95; do
    PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b${BETA}.parquet"
    OUT_DIR="${OUT_BASE}/b${BETA}"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "------------------------------------------------------------"
    echo "  beta_multi=${BETA}  →  ${OUT_DIR}"
    echo "------------------------------------------------------------"

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
        --weights "0.75,0.80,0.85" \
        --thresholds "0.075,0.090" \
        --topk 10 \
        --vit-aggregator max \
        --cnxt-aggregator max \
        --tag "exp14c_b${BETA}"

    # Tag CSVs so they don't collide on download
    for f in "${OUT_DIR}"/exp14c_b${BETA}_*.csv; do
        [ -e "$f" ] || continue
        base="$(basename "$f" .csv)"
        case "$base" in
            *_vitLK_cnxtK) ;;
            *) mv "$f" "${OUT_DIR}/${base}_vitLK_cnxtK.csv" ;;
        esac
    done
done

echo ""
echo "=== Stage 1b done ==="
echo "All 18 CSVs under ${OUT_BASE}/b{0.75,0.85,0.95}/"
echo ""
echo "Decision:"
echo "  - submit a representative subset to LB (e.g., one per beta at the most"
echo "    plausible (w, thr) combo, then expand around the winner)"
echo "  - if best CSV >= 0.456: proceed to stage 2 (fine sweep around winner)"
echo "  - if best CSV in [0.444, 0.456): marginal — stage 2 anyway, lower expectations"
echo "  - if best CSV  < 0.444: regression — diagnose before stage 2"
