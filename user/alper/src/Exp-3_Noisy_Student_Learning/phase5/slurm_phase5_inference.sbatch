#!/bin/bash
#SBATCH --job-name=p5-student-infer
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

TILE_ROOT="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/phase1_tiles_test"
CHECKPOINT="${PLANTCLEF_ROOT}/aerten3/phase4_checkpoints/run_kl_default/best_model.pt"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/phase5_student"
OUT_PARQUET="${OUT_DIR}/student_probs.parquet"
OUT_CSV="${OUT_DIR}/student_submission_top10_thr0.03.csv"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase5/student_inference.py \
  --tile-parquet-root "${TILE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --out-parquet "${OUT_PARQUET}" \
  --out-csv "${OUT_CSV}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --topk 10 \
  --threshold 0.03 \
  --aug-filter \
  --tile-filter-k 16 \
  --batch-size 64

echo "Done: Phase 5 student inference"
