#!/bin/bash
# Launcher for exp 003 — TileQ-Decoder training (GPU job, 80k steps, ~30-90 min).
#
# Usage:
#   cd user/murilo/experiments/003-tileq-decoder
#
#   bash sbatch/train.sh --sampler A                       # default seed=42 steps=80k
#   bash sbatch/train.sh --sampler B --seed 42
#   bash sbatch/train.sh --sampler A --gpu h100            # use H100 partition
#   bash sbatch/train.sh --sampler A --fast-dev-run        # smoke (200 steps, batch 8)
#   bash sbatch/train.sh --sampler A --no-wandb            # skip W&B (TB only)
#   bash sbatch/train.sh --sampler A --dry-run             # preview the sbatch command

set -eu

DRY_RUN=false
FAST_DEV_RUN=false
NO_WANDB=false
PROJECT_NAME="ps-clef2026_plantclef-0"
SAMPLER=""
SEED=42
STEPS=80000
BATCH_SIZE=32
GPU_TYPE=""

resolve_partitions() {
    local input="$1" result=""
    IFS=',' read -ra parts <<< "$input"
    for p in "${parts[@]}"; do
        p="$(echo "$p" | tr '[:upper:]' '[:lower:]' | xargs)"
        case "$p" in
            a100|gpu-a100)            p="gpu-a100" ;;
            h100|gpu-h100)            p="gpu-h100" ;;
            h200|gpu-h200)            p="gpu-h200" ;;
            rtx6000|rtx|gpu-rtx6000)  p="gpu-rtx6000" ;;
            v100|gpu-v100)            p="gpu-v100" ;;
            *) echo "Error: unknown GPU type '$p'" >&2; exit 1 ;;
        esac
        result="${result:+$result,}$p"
    done
    echo "$result"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --sampler)         SAMPLER="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --steps)           STEPS="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --gpu)             GPU_TYPE="$2"; shift 2 ;;
        --fast-dev-run)    FAST_DEV_RUN=true; shift ;;
        --no-wandb)        NO_WANDB=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --project-name)    PROJECT_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --sampler {A|B} [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --sampler {A|B}     Which sampler to train against"
            echo ""
            echo "Optional:"
            echo "  --seed N            Seed (default 42)"
            echo "  --steps N           Total steps (default 80000)"
            echo "  --batch-size N      Batch size (default 32)"
            echo "  --gpu TYPE          GPU partition: a100|h100|h200|rtx6000|v100"
            echo "                      (comma-separated for fallback; default rtx6000)"
            echo "  --fast-dev-run      200 steps, batch 8 — smoke test"
            echo "  --no-wandb          Skip W&B (TensorBoard still on)"
            echo "  --project-name N    PACE allocation (default ps-clef2026_plantclef-0)"
            echo "  --dry-run           Preview sbatch command without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$SAMPLER" ]]; then
    echo "Error: --sampler {A|B} is required" >&2; exit 1
fi
if [[ "$SAMPLER" != "A" && "$SAMPLER" != "B" ]]; then
    echo "Error: --sampler must be A or B, got '$SAMPLER'" >&2; exit 1
fi

if [[ -n "$GPU_TYPE" ]]; then
    PARTITION="$(resolve_partitions "$GPU_TYPE")"
else
    PARTITION="gpu-rtx6000"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_NAME="train_sampler${SAMPLER}_seed${SEED}"

echo "=============================================="
echo "Experiment 003: TileQ-Decoder training"
echo "=============================================="
echo "Sampler:       ${SAMPLER}"
echo "Seed:          ${SEED}"
echo "Steps:         ${STEPS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Partition:     ${PARTITION}"
echo "Fast dev run:  ${FAST_DEV_RUN}"
echo "W&B:           $([[ "${NO_WANDB}" == "true" ]] && echo "off" || echo "on")"
echo ""

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOB_SCRIPT="${SCRIPT_DIR}/train.sbatch"

CMD="sbatch \
    --partition=${PARTITION} \
    --output=${LOG_DIR}/slurm_${RUN_NAME}_%j.log \
    --export=ALL,PROJECT_NAME=${PROJECT_NAME},SAMPLER=${SAMPLER},SEED=${SEED},STEPS=${STEPS},BATCH_SIZE=${BATCH_SIZE},NO_WANDB=${NO_WANDB},FAST_DEV_RUN=${FAST_DEV_RUN} \
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
    echo "TB:     tensorboard --logdir ${EXPERIMENT_DIR}/artifacts/tensorboard"
fi
