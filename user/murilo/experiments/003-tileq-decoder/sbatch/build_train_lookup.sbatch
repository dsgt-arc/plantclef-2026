#!/bin/bash
#SBATCH --job-name=exp003-build-lookup
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_build_train_lookup_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 003: Build per-class-idx CLS lookup + species weights (spec §3.2-3).
#
# Reads 1.4 M ViT-B DINOv2 train CLS embeddings from
#   shared/embeddings/train/sample_id={0..19}/
# and writes:
#   artifacts/train_cls_by_class_idx.npz   (~4.3 GB, sorted-by-class embeddings)
#   artifacts/species_weights.npy          (7806-d float32, sums to 1)
#
# Usage (via build_train_lookup.sh launcher — preferred):
#   cd user/murilo/experiments/003-tileq-decoder
#   bash sbatch/build_train_lookup.sh
#
# Usage (direct sbatch — also supported):
#   sbatch sbatch/build_train_lookup.sbatch

set -eu

PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_plantclef-0}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

echo "=============================================="
echo "Job: Experiment 003 — Build train CLS lookup"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Experiment:    ${EXPERIMENT_DIR}"
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

echo "  Installing PyTorch with CUDA (forces downgrade from default-index torch)..."
uv pip install "torch>=2.1.0,<2.11" "torchvision>=0.16,<0.26" --index-url https://download.pytorch.org/whl/cu121

echo "  Virtual environment ready!"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import numpy, pyarrow; print(f'numpy: {numpy.__version__}, pyarrow: {pyarrow.__version__}')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="/storage/project/${PROJECT_NAME}/mgustineli3/cache/torch"
mkdir -p "$TORCH_HOME"

# ============================================================================
# Run the lookup build
# ============================================================================

mkdir -p "${EXPERIMENT_DIR}/logs"
mkdir -p "${EXPERIMENT_DIR}/artifacts"
cd "${EXPERIMENT_DIR}"

CMD="python build_train_lookup.py"
echo "Starting build..."
echo "  CMD: ${CMD}"
echo ""
srun ${CMD}

# ============================================================================
# Reload-and-validate tail (catch silent corruption before training reads it)
# ============================================================================

echo ""
echo "=== Output artefacts ==="
ls -lh "${EXPERIMENT_DIR}/artifacts/train_cls_by_class_idx.npz" \
       "${EXPERIMENT_DIR}/artifacts/species_weights.npy"

echo ""
echo "=== Reload sanity check ==="
python - <<PYEOF
import numpy as np

data = np.load("${EXPERIMENT_DIR}/artifacts/train_cls_by_class_idx.npz")
weights = np.load("${EXPERIMENT_DIR}/artifacts/species_weights.npy")

print(f"embeddings:          {data['embeddings'].shape} {data['embeddings'].dtype}")
print(f"offsets:             {data['offsets'].shape} {data['offsets'].dtype} "
      f"(first 5: {data['offsets'][:5].tolist()}, last: {int(data['offsets'][-1])})")
print(f"class_to_species_id: {data['class_to_species_id'].shape} "
      f"{data['class_to_species_id'].dtype}")
print(f"species_weights:     {weights.shape} {weights.dtype} sum={weights.sum():.6f}")

counts = np.diff(data['offsets'])
print(f"counts: min={counts.min()} median={int(np.median(counts))} max={counts.max()}")
assert (counts > 0).all(), "FAIL: classes with zero rows present"
assert abs(float(weights.sum()) - 1.0) < 1e-5, "FAIL: weights do not sum to 1"
assert data['embeddings'].shape == (1_408_033, 768), \
    f"FAIL: row count mismatch, got {data['embeddings'].shape}"
print("ALL CHECKS PASSED")
PYEOF

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
