#!/bin/bash
# Submit Phase 1 -> Phase 2 training chain for a single ConvNeXt variant.
# Usage: bash slurm/run_training.sh [base|tiny|small|large]

set -euo pipefail

VARIANT="${1:-base}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting training pipeline for ConvNeXt-${VARIANT}..."

# Submit Phase 1
P1_JOB=$(sbatch --export=VARIANT="${VARIANT}" "${SCRIPT_DIR}/phase1_train.sbatch" | awk '{print $4}')
echo "  Phase 1 job: ${P1_JOB} (head-only, ~1 hr)"

# Submit Phase 2 with dependency on Phase 1
P2_JOB=$(sbatch --export=VARIANT="${VARIANT}" --dependency=afterok:"${P1_JOB}" "${SCRIPT_DIR}/phase2_train.sbatch" | awk '{print $4}')
echo "  Phase 2 job: ${P2_JOB} (full fine-tune, ~6-10 hrs, depends on ${P1_JOB})"

echo "Done. Monitor with: squeue -u \$USER"
