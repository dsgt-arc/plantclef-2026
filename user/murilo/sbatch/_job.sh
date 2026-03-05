#!/bin/bash
#SBATCH --job-name=download-data
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=32:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Download PlantCLEF 2026 competition data (CPU-only)
#
# Usage:
#   cd user/murilo
#   bash sbatch/run.sh
#   bash sbatch/run.sh --dry-run
#   bash sbatch/run.sh --skip-extract
#

set -eu

# ============================================================================
# Determine paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    MURILO_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MURILO_DIR="$(dirname "$SCRIPT_DIR")"
fi

SCRIPTS_DIR="$MURILO_DIR/scripts"

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Download PlantCLEF 2026 Data"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Scripts dir: ${SCRIPTS_DIR}"
echo "Download args: ${DOWNLOAD_ARGS:-}"
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR
# ============================================================================

echo "Setting up per-job virtual environment..."

JOB_VENV_DIR="$TMPDIR/plantclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

# Keep uv cache on the same filesystem as the venv
export UV_CACHE_DIR="$TMPDIR/uv-cache"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing dependencies..."
uv pip install "$SCRIPTS_DIR"

echo "  Virtual environment ready!"
echo ""

# Verify aria2c is available
aria2c --version | head -1
echo ""

# ============================================================================
# Run download
# ============================================================================

echo "Starting download..."
echo ""

cd "$SCRIPTS_DIR"
python download_data.py ${DOWNLOAD_ARGS:-}

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "=============================================="
