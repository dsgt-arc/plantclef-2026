#!/bin/bash
# Launcher script for data download job.
#
# Usage:
#   cd user/murilo
#   bash sbatch/run.sh
#   bash sbatch/run.sh --skip-extract
#   bash sbatch/run.sh --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

SBATCH_DRY_RUN=false
DOWNLOAD_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            SBATCH_DRY_RUN=true
            shift
            ;;
        --skip-extract)
            DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --skip-extract"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-extract  Download only, skip tar extraction"
            echo "  --dry-run       Preview sbatch command without submitting"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Determine paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MURILO_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Submit job
# ============================================================================

echo "=============================================="
echo "PlantCLEF 2026: Download Data"
echo "=============================================="
echo "Murilo dir: ${MURILO_DIR}"
echo "Download args: ${DOWNLOAD_ARGS:-<none>}"
echo ""

LOG_DIR="${MURILO_DIR}/sbatch/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

CMD="sbatch \
    --output=${LOG_DIR}/slurm_download_%j.log \
    --export=ALL,DOWNLOAD_ARGS=${DOWNLOAD_ARGS} \
    ${JOB_SCRIPT}"

if [[ "${SBATCH_DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    cd "$MURILO_DIR"
    ${CMD}
    echo ""
    echo "Job submitted!"
fi

echo ""
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo "View logs in: ${LOG_DIR}/"
echo "=============================================="
