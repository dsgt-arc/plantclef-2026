#!/bin/bash
# Launcher script for Experiment 003 TileQ-Decoder — Test-CLS extraction.
#
# Usage:
#   cd user/murilo/experiments/003-tileq-decoder
#
#   bash sbatch/extract_test_cls.sh                   # full run (~2 GPU-h on RTX 6000)
#   bash sbatch/extract_test_cls.sh --fast-dev-run    # 5-image smoke test (~5 min)
#   bash sbatch/extract_test_cls.sh --gpu h100        # use H100 partition instead of RTX 6000
#   bash sbatch/extract_test_cls.sh --num-workers 16  # override DataLoader workers
#   bash sbatch/extract_test_cls.sh --dry-run         # preview the sbatch command

set -eu

DRY_RUN=false
FAST_DEV_RUN=false
NUM_WORKERS=""
PROJECT_NAME="ps-clef2026_plantclef-0"
GPU_TYPE=""

# Map short names to SLURM partition names.
resolve_partitions() {
    local input="$1"
    local result=""
    IFS=',' read -ra parts <<< "$input"
    for p in "${parts[@]}"; do
        p="$(echo "$p" | tr '[:upper:]' '[:lower:]' | xargs)"
        case "$p" in
            a100|gpu-a100)            p="gpu-a100" ;;
            h100|gpu-h100)            p="gpu-h100" ;;
            h200|gpu-h200)            p="gpu-h200" ;;
            rtx6000|rtx|gpu-rtx6000)  p="gpu-rtx6000" ;;
            v100|gpu-v100)            p="gpu-v100" ;;
            *)
                echo "Error: unknown GPU type '$p'. Options: a100, h100, h200, rtx6000, v100" >&2
                exit 1
                ;;
        esac
        result="${result:+$result,}$p"
    done
    echo "$result"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --fast-dev-run)  FAST_DEV_RUN=true; shift ;;
        --gpu)           GPU_TYPE="$2"; shift 2 ;;
        --num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --project-name)  PROJECT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit the test-CLS extraction job (exp 003 Priority 3)."
            echo ""
            echo "Options:"
            echo "  --gpu TYPE        GPU partition: a100, h100, h200, rtx6000, v100"
            echo "                    Comma-separated for fallback: --gpu rtx6000,v100"
            echo "                    Default: gpu-rtx6000"
            echo "  --fast-dev-run    Process 5 images only, skip parquet save"
            echo "  --num-workers N   Override DataLoader workers (default: 8)"
            echo "  --project-name N  PACE allocation (default: ps-clef2026_plantclef-0)"
            echo "  --dry-run         Preview sbatch command without submitting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Resolve GPU partition.
if [[ -n "$GPU_TYPE" ]]; then
    PARTITION="$(resolve_partitions "$GPU_TYPE")"
else
    PARTITION="gpu-rtx6000"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="extract_test_cls"

echo "=============================================="
echo "Experiment 003: Test-CLS extraction (grid 4×4)"
echo "=============================================="
echo "Partition:    ${PARTITION}"
echo "Mode:         $([[ "${FAST_DEV_RUN}" == "true" ]] && echo "fast-dev-run (5 images, no parquet)" || echo "full run (2,105 images → cls_grid_4x4.parquet)")"
echo "Output:       /storage/project/${PROJECT_NAME}/shared/inference/dinov2-vit-base/cls_grid_4x4.parquet"
[[ -n "$NUM_WORKERS" ]] && echo "Num workers:  ${NUM_WORKERS}"
echo ""

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/extract_test_cls.sbatch"

CPUS_OVERRIDE=""
if [[ -n "$NUM_WORKERS" ]]; then
    CPUS_OVERRIDE="--cpus-per-task=${NUM_WORKERS}"
fi

CMD="sbatch \
    --partition=${PARTITION} \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,PROJECT_NAME=${PROJECT_NAME},FAST_DEV_RUN=${FAST_DEV_RUN},NUM_WORKERS=${NUM_WORKERS} \
    --job-name=exp003-${RUN_NAME} \
    ${CPUS_OVERRIDE} \
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
