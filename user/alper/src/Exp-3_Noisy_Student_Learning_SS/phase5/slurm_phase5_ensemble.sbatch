#!/bin/bash
#SBATCH --job-name=p5-ensemble
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aerten3@gatech.edu

# Phase 5: Ensemble student + teacher predictions
# Sweeps alpha values: 0.3, 0.5, 0.7, 0.9
# CPU-only — no GPU needed (just loads parquets and blends)

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

export PYTHONPATH="/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning:${PYTHONPATH:-}"

mkdir -p logs

source /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

# Teacher probs: Phase 2 sweep best config (T=2.0, k=6)
TEACHER_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/phase2_sweep/sweep_T2.0_k6.parquet"

# Student probs: Phase 5 combined baseline output
STUDENT_PARQUET="${PLANTCLEF_ROOT}/aerten3/submissions/phase5_student/student_probs.parquet"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/phase5_ensemble"

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-3_Noisy_Student_Learning/phase5/ensemble_inference.py \
  --teacher-parquet "${TEACHER_PARQUET}" \
  --student-parquet "${STUDENT_PARQUET}" \
  --out-dir "${OUT_DIR}" \
  --species-map "${SPECIES_MAP}" \
  --geo-mask-npy "${GEO_MASK}" \
  --alphas 0.3,0.5,0.7,0.9 \
  --topk 10 \
  --threshold 0.03 \
  --save-parquet

echo "Done: Phase 5 ensemble (4 alpha values)"
