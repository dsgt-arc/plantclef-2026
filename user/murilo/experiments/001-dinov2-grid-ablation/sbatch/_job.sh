#!/bin/bash
#SBATCH --job-name=exp001-grid-ablation
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 001: DINOv2 Grid-Size Ablation on PlantCLEF 2026
# Stage 1: GPU inference → parquet (see inference.py)
#
# Usage (via run.sh):
#   bash sbatch/run.sh --grid-size 3
#   bash sbatch/run.sh --all
#   bash sbatch/run.sh --grid-size 4 --fast-dev-run
#

set -eu

# Environment variables passed from run.sh
GRID_SIZE="${GRID_SIZE:-0}"
PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_plantclef-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
GPU_CHECK="${GPU_CHECK:-0}"

# ============================================================================
# Determine paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Experiment 001 - DINOv2 Grid Ablation"
echo "  Stage 1: GPU Inference → Parquet"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "Grid size:    ${GRID_SIZE}"
echo "Project:      ${PROJECT_NAME}"
echo "Experiment:   ${EXPERIMENT_DIR}"
echo ""
nvidia-smi
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

# Per-job uv cache on same filesystem as venv (avoids hardlink issues)
export UV_CACHE_DIR="$TMPDIR/uv-cache"

# Job-specific venv in TMPDIR for fast local I/O
JOB_VENV_DIR="$TMPDIR/plantclef-${SLURM_JOB_ID}"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv:   $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11

# Activate
source "$JOB_VENV/bin/activate"

# Install experiment dependencies
echo "  Installing dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

# Install PyTorch with CUDA support
echo "  Installing PyTorch with CUDA..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
echo ""

# ============================================================================
# Environment variables
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model/data cache in project storage (persistent)
export TORCH_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/torch"
mkdir -p "$TORCH_HOME"

# ============================================================================
# Run inference (Stage 1)
# ============================================================================

cd "$EXPERIMENT_DIR"
mkdir -p logits logs

echo "Starting inference..."
echo ""

INFERENCE_CMD="python inference.py --grid-size ${GRID_SIZE}"

if [[ "${FAST_DEV_RUN}" == "true" ]]; then
    INFERENCE_CMD="${INFERENCE_CMD} --fast-dev-run"
    echo "  (fast-dev-run: 5 images, no parquet output)"
fi
if [[ "${GPU_CHECK}" -gt 0 ]]; then
    INFERENCE_CMD="${INFERENCE_CMD} --gpu-check ${GPU_CHECK}"
    echo "  (gpu-check: ${GPU_CHECK} images, profiling mode)"
fi

srun ${INFERENCE_CMD}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
