#!/bin/bash
# Exp-11B: blend Exp-9B ViT+kNN+ArcFace fused probabilities with ConvNeXt
# probabilities. Same pipeline as Exp-11's best (Approach A) but with the
# stronger ViT branch from Exp-9B Tier 2.

#SBATCH --job-name=exp11b-ens-knn-arcface
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail
export PYTHONUNBUFFERED=1

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"

EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"
EXP7_DIR="${REPO_ROOT}/user/alper/src/Exp-7_Multiscale_tiling"

export PYTHONPATH="${EXP7_DIR}:${PYTHONPATH:-}"

mkdir -p logs

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"

# === ViT branch: Exp-9B Tier 2 fused parquet at beta_multi=0.85, b_single=1.00, alpha=0.15 ===
# This is the F1=0.42267 leaderboard config. The fused parquet was produced by
# slurm_diff_beta_phase2a.sbatch -> multiscale_aggregate_knn.py.
VIT_KNN_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_fused_a0.15.parquet"

# === ConvNeXt branch: reused unchanged from Exp-11 ===
CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"
CNXT_SCALES="${CNXT_DIR}/scale_4x4.parquet ${CNXT_DIR}/scale_5x5.parquet ${CNXT_DIR}/scale_6x6.parquet"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11b_ensemble_arcface_knn"
mkdir -p "${OUT_DIR}"

echo "=== Exp-11B: ConvNeXt + ViT+kNN+ArcFace ensemble ==="
echo "ViT branch:    ${VIT_KNN_PARQUET}"
echo "ConvNeXt dir:  ${CNXT_DIR}"
echo "Output:        ${OUT_DIR}"
echo ""
echo "Sanity checks:"
ls -lh "${VIT_KNN_PARQUET}" || { echo "ERROR: ViT+kNN parquet not found"; exit 1; }
ls -lh "${CNXT_DIR}"/*.parquet || { echo "ERROR: ConvNeXt parquets not found"; exit 1; }

echo ""
echo "=== Running weight sweep (cnxt_alpha=0.30, thr=0.100) ==="
echo "Exp-11 baseline: w=0.77 -> F1=0.4379 (with old ViT+kNN at F1=0.41754)"
echo "Exp-11B target:  beat 0.4409 (+0.003 over Exp-11)"
echo ""

python "${EXP11_DIR}/ensemble_with_knn.py" \
    --vit-knn-parquet "${VIT_KNN_PARQUET}" \
    --convnext-scales ${CNXT_SCALES} \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --cnxt-alpha 0.30 \
    --weights "0.74,0.76,0.77,0.78,0.80,0.82" \
    --thresholds "0.100" \
    --topk 10

echo ""
echo "=== Done ==="
echo "CSVs in: ${OUT_DIR}"
ls -1 "${OUT_DIR}"/*.csv
