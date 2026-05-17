#!/bin/bash
#SBATCH --job-name=exp003-train
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_train_%j.log
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 003: Train the TileQ-Decoder (spec §4).
#
# Required env vars (set by train.sh launcher; --export=ALL):
#   SAMPLER     A | B
#   SEED        e.g. 42
#   STEPS       e.g. 80000
#   BATCH_SIZE  e.g. 32
#   PROJECT_NAME (default ps-clef2026_plantclef-0)
#
# Optional:
#   NO_WANDB    "true" to skip W&B login (otherwise ensure WANDB_API_KEY is set)
#   FAST_DEV_RUN "true" -> quick 200-step run with batch 8 for smoke testing
#
# Usage (via train.sh launcher — preferred):
#   bash sbatch/train.sh --sampler A --seed 42
#   bash sbatch/train.sh --sampler B --seed 42 --gpu h100

set -eu

PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_plantclef-0}"
SAMPLER="${SAMPLER:?SAMPLER env var must be A or B}"
SEED="${SEED:-42}"
STEPS="${STEPS:-80000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NO_WANDB="${NO_WANDB:-false}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

if [[ "${FAST_DEV_RUN}" == "true" ]]; then
    STEPS=200
    BATCH_SIZE=8
fi

echo "=============================================="
echo "Job: Experiment 003 — TileQ-Decoder training"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Sampler:       ${SAMPLER}"
echo "Seed:          ${SEED}"
echo "Steps:         ${STEPS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Fast dev run:  ${FAST_DEV_RUN}"
echo ""
nvidia-smi
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR
# ============================================================================

echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/plantclef-${SLURM_JOB_ID}"
JOB_VENV="$JOB_VENV_DIR/.venv"
mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing exp 003 dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing PyTorch with CUDA (forces downgrade from default-index torch)..."
uv pip install "torch>=2.1.0,<2.11" "torchvision>=0.16,<0.26" --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, bf16: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None}')"
python -c "import wandb, tensorboard; print(f'wandb {wandb.__version__}, tensorboard {tensorboard.__version__}')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/torch"
mkdir -p "$TORCH_HOME"

# ============================================================================
# Sanity gate: run the overfit-one-batch + shape audit before training
# ============================================================================

mkdir -p "${EXPERIMENT_DIR}/logs"
mkdir -p "${EXPERIMENT_DIR}/artifacts/checkpoints"
cd "${EXPERIMENT_DIR}"

echo "=== Sanity gate: python sanity_check.py ==="
python sanity_check.py --steps 100 --overfit-ratio 5.0
echo ""

# ============================================================================
# Train
# ============================================================================

WANDB_FLAG=""
if [[ "${NO_WANDB}" == "true" ]]; then
    WANDB_FLAG="--no-wandb"
fi

CMD="python train.py --sampler ${SAMPLER} --seed ${SEED} --steps ${STEPS} --batch-size ${BATCH_SIZE} ${WANDB_FLAG}"
echo "Starting training..."
echo "  CMD: ${CMD}"
echo ""
srun ${CMD}

# ============================================================================
# Final artefact listing
# ============================================================================

echo ""
echo "=== Output artefacts ==="
ls -lh "${EXPERIMENT_DIR}/artifacts/checkpoints/decoder_sampler${SAMPLER}.pt" 2>/dev/null || true
ls -lh "${EXPERIMENT_DIR}/artifacts/loss_sampler${SAMPLER}."{csv,png} 2>/dev/null || true
ls -lh "${EXPERIMENT_DIR}/artifacts/config_sampler${SAMPLER}.yaml" 2>/dev/null || true

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
