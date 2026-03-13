#!/bin/bash
# Launcher script for Experiment 001: DINOv2 Grid-Size Ablation.
# Submits Stage 1 (GPU inference → parquet) jobs via sbatch.
#
# Usage:
#   cd user/murilo/experiments/001-dinov2-grid-ablation
#
#   # No grid (full image inference)
#   bash sbatch/run.sh
#   bash sbatch/run.sh --grid-size 0
#
#   # Specific grid size
#   bash sbatch/run.sh --grid-size 3
#   bash sbatch/run.sh --grid-size 4
#
#   # Submit all grid sizes (0, 2, 3, 4, ..., 10) as separate jobs
#   bash sbatch/run.sh --all
#
#   # Dev/profiling modes
#   bash sbatch/run.sh --grid-size 3 --fast-dev-run
#   bash sbatch/run.sh --grid-size 3 --gpu-check 20
#   bash sbatch/run.sh --all --dry-run
#

set -eu

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=false
FAST_DEV_RUN=false
GPU_CHECK=0
GRID_SIZE=0
ALL=false
PROJECT_NAME="ps-clef2026_plantclef-0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL=true
            shift
            ;;
        --grid-size)
            GRID_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --fast-dev-run)
            FAST_DEV_RUN=true
            shift
            ;;
        --gpu-check)
            GPU_CHECK="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit Stage 1 (GPU inference → parquet) jobs."
            echo "Run Stage 2 (classification) separately with classify.py."
            echo ""
            echo "Options:"
            echo "  --grid-size N       Grid size for NxN tiling (default: 0 = no grid)"
            echo "  --all               Submit jobs for all grid sizes (0, 2, 3, ..., 10)"
            echo "  --fast-dev-run      Process 5 images only (pipeline validation)"
            echo "  --gpu-check N       Process N images for GPU profiling"
            echo "  --project-name NAME PACE project allocation (default: ps-clef2026_plantclef-0)"
            echo "  --dry-run           Preview sbatch command without submitting"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# --all: submit one job per grid size
# ============================================================================

if [[ "${ALL}" == "true" ]]; then
    EXTRA_ARGS=()
    [[ "${DRY_RUN}" == "true" ]] && EXTRA_ARGS+=(--dry-run)
    [[ "${FAST_DEV_RUN}" == "true" ]] && EXTRA_ARGS+=(--fast-dev-run)
    [[ "${GPU_CHECK}" -gt 0 ]] && EXTRA_ARGS+=(--gpu-check "${GPU_CHECK}")
    [[ "${PROJECT_NAME}" != "ps-clef2026_plantclef-0" ]] && EXTRA_ARGS+=(--project-name "${PROJECT_NAME}")

    # Grid sizes: 0 (no grid), 2, 3, 4, 5, 6, 7, 8, 9, 10
    for gs in 0 2 3 4 5 6 7 8 9 10; do
        echo ">>> Submitting grid-size=${gs}..."
        bash "${BASH_SOURCE[0]}" --grid-size "${gs}" "${EXTRA_ARGS[@]}"
        echo ""
    done
    exit 0
fi

# ============================================================================
# Determine paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Setup
# ============================================================================

if [[ "${GRID_SIZE}" -eq 0 ]]; then
    GRID_LABEL="no_grid"
else
    GRID_LABEL="grid_${GRID_SIZE}x${GRID_SIZE}"
fi

echo "=============================================="
echo "Experiment 001: DINOv2 Grid Ablation"
echo "  Stage 1: GPU Inference → Parquet"
echo "=============================================="
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Grid size:      ${GRID_LABEL}"
echo "Project:        ${PROJECT_NAME}"
echo "Fast dev run:   ${FAST_DEV_RUN}"
echo "GPU check:      ${GPU_CHECK}"
echo ""

# Create log directory
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/_job.sh"

# ============================================================================
# Submit job
# ============================================================================

EXPORT_VARS="ALL,GRID_SIZE=${GRID_SIZE},PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},GPU_CHECK=${GPU_CHECK}"

CMD="sbatch \
    --output=${LOG_DIR}/slurm_${GRID_LABEL}_%j.log \
    --export=${EXPORT_VARS} \
    --job-name=exp001-${GRID_LABEL} \
    ${JOB_SCRIPT}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${CMD}"
else
    echo "Submitting job..."
    cd "$EXPERIMENT_DIR"
    ${CMD}
    echo ""
    echo "=============================================="
    echo "Job submitted!"
    echo "=============================================="
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "View logs in: ${LOG_DIR}/"
    echo ""
    echo "After job completes, run Stage 2 (exp 002):"
    echo "  python ../002-classification-methods/classify.py --input logits/${GRID_LABEL}.parquet --method expanded-list --top-k 5 --max-species 20"
fi
