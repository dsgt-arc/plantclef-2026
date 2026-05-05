#!/bin/bash
# Launcher for exp 003 — FAISS background-cleaning map build (CPU job).
#
# Usage:
#   cd user/murilo/experiments/003-tileq-decoder
#
#   bash sbatch/build_faiss_background_map.sh                    # default tau=0.85
#   bash sbatch/build_faiss_background_map.sh --threshold 0.80   # override
#   bash sbatch/build_faiss_background_map.sh --dry-run

set -eu

DRY_RUN=false
PROJECT_NAME="ps-clef2026_plantclef-0"
THRESHOLD="0.85"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --threshold)     THRESHOLD="$2"; shift 2 ;;
        --project-name)  PROJECT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit the FAISS background-cleaning map build job (CPU, ~10-15 min)."
            echo ""
            echo "Options:"
            echo "  --threshold T     Cosine-sim cutoff (spec default 0.85)"
            echo "  --project-name N  PACE allocation (default: ps-clef2026_plantclef-0)"
            echo "  --dry-run         Preview sbatch command without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="build_faiss_background_map"

echo "=============================================="
echo "Experiment 003: Build FAISS background map"
echo "=============================================="
echo "Threshold:    ${THRESHOLD}"
echo "Output:       ${EXPERIMENT_DIR}/artifacts/sampler_b_lookup.npz + histogram + summary"
echo ""

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/build_faiss_background_map.sbatch"

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,PROJECT_NAME=${PROJECT_NAME},THRESHOLD=${THRESHOLD} \
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
