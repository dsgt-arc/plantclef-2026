#!/bin/bash
# Submit Phase 1 -> Phase 2 training chain for a single ConvNeXt variant.
#
# Usage:
#   bash slurm/run_training.sh base                    # 1x H200 (default)
#   bash slurm/run_training.sh base h100 4             # 4x H100 for Phase 2
#   bash slurm/run_training.sh large h100 2             # 2x H100 for Phase 2

set -euo pipefail

VARIANT="${1:-base}"
GPU_TYPE="${2:-h200}"
NGPUS="${3:-1}"
GRAD_ACCUM="${4:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve partition and GRES for Phase 2
case "${GPU_TYPE}" in
    h200) PARTITION="gpu-h200"; GRES="gpu:h200:${NGPUS}" ;;
    h100) PARTITION="gpu-h100"; GRES="gpu:h100:${NGPUS}" ;;
    a100) PARTITION="gpu-a100"; GRES="gpu:a100:${NGPUS}" ;;
    *)    echo "Unknown GPU type: ${GPU_TYPE}"; exit 1 ;;
esac

CPUS=$((NGPUS * 8))
MEM=$((NGPUS * 64))

echo "Submitting training pipeline for ConvNeXt-${VARIANT}..."
echo "  Phase 2 GPU: ${NGPUS}x ${GPU_TYPE}, grad_accum=${GRAD_ACCUM}"

# Submit Phase 1 (always 1x H200, fast head-only training)
P1_JOB=$(sbatch --export=VARIANT="${VARIANT}" "${SCRIPT_DIR}/phase1_train.sbatch" | awk '{print $4}')
echo "  Phase 1 job: ${P1_JOB} (1x H200, head-only, ~1-2 hrs)"

# Submit Phase 2 with chosen GPU config
P2_JOB=$(sbatch \
    --partition="${PARTITION}" \
    --gres="${GRES}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}G" \
    --export=VARIANT="${VARIANT}",GPU_TYPE="${GPU_TYPE}",NGPUS="${NGPUS}",GRAD_ACCUM="${GRAD_ACCUM}" \
    --dependency=afterok:"${P1_JOB}" \
    "${SCRIPT_DIR}/phase2_train.sbatch" | awk '{print $4}')
echo "  Phase 2 job: ${P2_JOB} (${NGPUS}x ${GPU_TYPE}, depends on ${P1_JOB})"

echo "Done. Monitor with: squeue -u \$USER"
