#!/bin/bash
#SBATCH --job-name=exp003-extract-cls
#SBATCH --account=paceship-clef2026_plantclef
# --partition is set by run.sh (default: gpu-rtx6000; override with --gpu)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 003: TileQ-Decoder — Test-CLS extraction (Priority 3, spec §3.1).
#
# Runs inference.py from exp 001 with --save-cls --grid-size 4 against the
# 2,105-image PlantCLEF 2026 test set. Output:
#   shared/inference/dinov2-vit-base/cls_grid_4x4.parquet
#   schema: image_name (str), tile_idx (int32), cls_embedding (binary float16[768])
#   expected rows: 33,680 (2,105 × 16)
#
# Usage (via run.sh):
#   cd user/murilo/experiments/003-tileq-decoder
#   bash sbatch/run.sh --fast-dev-run    # 5-image smoke test
#   bash sbatch/run.sh                   # full run (~2 GPU-h on RTX 6000)

set -eu

PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_plantclef-0}"
FAST_DEV_RUN="${FAST_DEV_RUN:-false}"
NUM_WORKERS="${NUM_WORKERS:-}"
GRID_SIZE=4

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

USER_DIR="$(dirname "$EXPERIMENT_DIR")"
EXP001_DIR="$USER_DIR/001-dinov2-grid-ablation"
OUTPUT_DIR="/storage/project/${PROJECT_NAME}/shared/inference/dinov2-vit-base"
OUTPUT_PATH="${OUTPUT_DIR}/cls_grid_${GRID_SIZE}x${GRID_SIZE}.parquet"

echo "=============================================="
echo "Job: Experiment 003 — Test-CLS extraction"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Experiment:    ${EXPERIMENT_DIR}"
echo "Inference src: ${EXP001_DIR}"
echo "Output:        ${OUTPUT_PATH}"
echo "Fast dev run:  ${FAST_DEV_RUN}"
echo ""
nvidia-smi
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."
export UV_CACHE_DIR="$TMPDIR/uv-cache"
JOB_VENV_DIR="$TMPDIR/plantclef-${SLURM_JOB_ID}"
JOB_VENV="$JOB_VENV_DIR/.venv"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing exp 003 dependencies..."
uv pip install -e "$EXPERIMENT_DIR"

echo "  Installing PyTorch with CUDA..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import pyarrow; print(f'pyarrow: {pyarrow.__version__}')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/torch"
mkdir -p "$TORCH_HOME"

# ============================================================================
# Run CLS extraction (cd to exp 001 so the script's relative paths work)
# ============================================================================

mkdir -p "$EXPERIMENT_DIR/logs"
cd "$EXP001_DIR"
mkdir -p logs

INFERENCE_CMD="python inference.py --save-cls --grid-size ${GRID_SIZE} --output-dir ${OUTPUT_DIR}"
if [[ "${FAST_DEV_RUN}" == "true" ]]; then
    INFERENCE_CMD="${INFERENCE_CMD} --fast-dev-run"
    echo "  (fast-dev-run: 5 images, no parquet output)"
fi
if [[ -n "${NUM_WORKERS}" ]]; then
    INFERENCE_CMD="${INFERENCE_CMD} --num-workers ${NUM_WORKERS}"
fi

echo "Starting CLS extraction..."
echo "  CMD: ${INFERENCE_CMD}"
echo ""
srun ${INFERENCE_CMD}

# ============================================================================
# Validation tail (Task 2.4): row count + per-image-group sanity check.
# Skipped in fast-dev-run since no parquet is written.
# ============================================================================

if [[ "${FAST_DEV_RUN}" != "true" ]]; then
    echo ""
    echo "=== Validation: row count + per-image groups ==="
    python - <<PYEOF
import pyarrow.parquet as pq
import pyarrow.compute as pc

path = "${OUTPUT_PATH}"
expected_images = 2105
expected_tiles_per_image = ${GRID_SIZE} * ${GRID_SIZE}
expected_rows = expected_images * expected_tiles_per_image  # 33,680

table = pq.read_table(path)
n_rows = table.num_rows
unique_images = pc.unique(table["image_name"]).length()

print(f"Path:          {path}")
print(f"Rows:          {n_rows} (expected {expected_rows})")
print(f"Unique images: {unique_images} (expected {expected_images})")
print(f"Schema:        {table.schema}")

assert n_rows == expected_rows, f"row count mismatch: {n_rows} != {expected_rows}"
assert unique_images == expected_images, f"image count mismatch: {unique_images} != {expected_images}"
assert n_rows == unique_images * expected_tiles_per_image, "per-image tile count mismatch"
print("Validation PASSED")
PYEOF
fi

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
