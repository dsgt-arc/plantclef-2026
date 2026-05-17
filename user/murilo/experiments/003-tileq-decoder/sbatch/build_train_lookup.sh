#!/bin/bash
# Launcher for exp 003 — train CLS lookup build (CPU job, ~5 min wallclock).
#
# Usage:
#   cd user/murilo/experiments/003-tileq-decoder
#
#   bash sbatch/build_train_lookup.sh             # submit full build
#   bash sbatch/build_train_lookup.sh --dry-run   # preview the sbatch command

set -eu

DRY_RUN=false
PROJECT_NAME="ps-clef2026_plantclef-0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --project-name)  PROJECT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit the train CLS lookup build job (CPU, ~5 min)."
            echo ""
            echo "Options:"
            echo "  --project-name N  PACE allocation (default: ps-clef2026_plantclef-0)"
            echo "  --dry-run         Preview sbatch command without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="build_train_lookup"

echo "=============================================="
echo "Experiment 003: Build train CLS lookup"
echo "=============================================="
echo "Output:       ${EXPERIMENT_DIR}/artifacts/{train_cls_by_class_idx.npz, species_weights.npy}"
echo ""

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/build_train_lookup.sbatch"

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,PROJECT_NAME=${PROJECT_NAME} \
    --job-name=exp003-${RUN_NAME} \
    ${JOB_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    ${CMD}
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Logs:    ${LOG_DIR}/"
fi
