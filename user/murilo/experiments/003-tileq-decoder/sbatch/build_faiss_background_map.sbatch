#!/bin/bash
#SBATCH --job-name=exp003-build-faiss
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_build_faiss_background_map_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 003: Build FAISS background-cleaning map for Sampler B (spec §3.4).
#
# Reads:
#   artifacts/train_cls_by_class_idx.npz                     (~4.3 GB)
#   shared/inference/dinov2-vit-base/cls_grid_4x4.parquet    (~50 MB)
# Writes:
#   artifacts/sampler_b_lookup.npz       (~100 MB: image_names, test_cls,
#                                          top1_sim, top1_class_idx,
#                                          tile_to_class_idx, threshold)
#   artifacts/tile_top1_sim_histogram.png
#   artifacts/faiss_threshold.txt        (percentile table + label coverage)
#
# Memory peak ~13 GB (4.3 train npz + 4.3 train copy + 4.3 FAISS internal),
# fits comfortably in the 48 GB request.
#
# Usage (via launcher):
#   bash sbatch/build_faiss_background_map.sh
#   bash sbatch/build_faiss_background_map.sh --threshold 0.85   # override tau
#
# Usage (direct sbatch):
#   sbatch sbatch/build_faiss_background_map.sbatch

set -eu

PROJECT_NAME="${PROJECT_NAME:-ps-clef2026_plantclef-0}"
THRESHOLD="${THRESHOLD:-0.85}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

# Direct sbatch invocation can pass `--threshold X` as positional args.
for arg in "$@"; do
    case "$arg" in
        --threshold=*) THRESHOLD="${arg#*=}" ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo "Job: Experiment 003 — Build FAISS background map"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Experiment:    ${EXPERIMENT_DIR}"
echo "Threshold:     ${THRESHOLD}"
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (matches extract_test_cls pattern)
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

python -c "import faiss, numpy, pyarrow, matplotlib; print(f'faiss {faiss.__version__}, numpy {numpy.__version__}, pyarrow {pyarrow.__version__}')"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# Run the FAISS map build
# ============================================================================

mkdir -p "${EXPERIMENT_DIR}/logs"
mkdir -p "${EXPERIMENT_DIR}/artifacts"
cd "${EXPERIMENT_DIR}"

CMD="python build_faiss_background_map.py --threshold ${THRESHOLD}"
echo "Starting build..."
echo "  CMD: ${CMD}"
echo ""
srun ${CMD}

# ============================================================================
# Reload-and-validate tail
# ============================================================================

echo ""
echo "=== Output artefacts ==="
ls -lh "${EXPERIMENT_DIR}/artifacts/sampler_b_lookup.npz" \
       "${EXPERIMENT_DIR}/artifacts/tile_top1_sim_histogram.png" \
       "${EXPERIMENT_DIR}/artifacts/faiss_threshold.txt"

echo ""
echo "=== Reload sanity check ==="
python - <<PYEOF
import numpy as np

data = np.load("${EXPERIMENT_DIR}/artifacts/sampler_b_lookup.npz", allow_pickle=False)
expected = {"image_names", "test_cls", "top1_sim", "top1_class_idx",
            "tile_to_class_idx", "threshold"}
got = set(data.files)
assert got == expected, f"FAIL: missing/extra keys {got ^ expected}"

print(f"image_names:       {data['image_names'].shape} {data['image_names'].dtype}")
print(f"test_cls:          {data['test_cls'].shape} {data['test_cls'].dtype}")
print(f"top1_sim:          {data['top1_sim'].shape} {data['top1_sim'].dtype}")
print(f"top1_class_idx:    {data['top1_class_idx'].shape} {data['top1_class_idx'].dtype}")
print(f"tile_to_class_idx: {data['tile_to_class_idx'].shape} {data['tile_to_class_idx'].dtype}")
print(f"threshold:         {float(data['threshold']):.4f}")

# Schema checks
assert data["image_names"].shape == (2105,), \
    f"FAIL: expected 2105 image_names, got {data['image_names'].shape}"
assert data["test_cls"].shape == (2105, 16, 768), \
    f"FAIL: expected (2105, 16, 768), got {data['test_cls'].shape}"
assert data["top1_sim"].shape == (2105, 16)
assert data["top1_class_idx"].shape == (2105, 16)
assert data["tile_to_class_idx"].shape == (2105, 16)

# Distribution sanity: top1_sim ought to be in [-1, 1] (cosine), mostly positive
sim = data["top1_sim"]
assert sim.min() >= -1.001 and sim.max() <= 1.001, \
    f"FAIL: top1_sim out of [-1, 1] range: min={sim.min()} max={sim.max()}"

# Class indices in valid range or -1
ttc = data["tile_to_class_idx"]
mapped = ttc[ttc >= 0]
assert mapped.min() >= 0 and mapped.max() < 7806, \
    f"FAIL: class_idx out of [0, 7806): min={mapped.min()} max={mapped.max()}"

# Coverage at the chosen threshold
labelled = (ttc >= 0).sum()
total = ttc.size
print(f"\nLabel coverage at threshold {float(data['threshold']):.2f}: "
      f"{labelled}/{total} ({100*labelled/total:.1f}%)")

print("ALL CHECKS PASSED")
PYEOF

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
