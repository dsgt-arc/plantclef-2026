#!/bin/bash
#
# Create and set up virtual environment for PlantCLEF 2026 experiments
# Run this in an interactive session to set up the environment
#
# This script:
#   1. Sets up environment variables
#   2. Installs uv package manager
#   3. Creates virtual environment in /tmp (fast local I/O)
#   4. Installs dependencies from the target directory's pyproject.toml
#
# Usage:
#   salloc -p gpu-rtx6000 --gres=gpu:1 -c 4 --mem=32G -t 2:00:00
#   bash create_venv.sh --exp-num 1
#   bash create_venv.sh --dir scripts
#
# Options:
#   --exp-num NUM  Experiment number (e.g., 1, 2, 3)
#   --dir NAME     Directory name relative to user/murilo/ (e.g., scripts)
#   --no-gpu       Skip GPU check (for CPU-only tasks like data download)
#   -h, --help     Show this help message
#
# One of --exp-num or --dir is required.
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Argument parsing
# ============================================================================

EXP_NUM=""
DIR_NAME=""
REQUIRE_GPU=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: bash create_venv.sh [--exp-num NUM | --dir NAME] [--no-gpu]"
    echo ""
    echo "Options:"
    echo "  --exp-num NUM  Experiment number (e.g., 1, 2, 3)"
    echo "  --dir NAME     Directory name relative to user/murilo/ (e.g., scripts)"
    echo "  --no-gpu       Skip GPU check (for CPU-only tasks like data download)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Available experiments:"
    for d in "$SCRIPT_DIR"/experiments/[0-9][0-9][0-9]-*/; do
        [ -d "$d" ] && echo "                   $(basename "$d")"
    done
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-num)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --exp-num requires a value"
                usage 1
            fi
            EXP_NUM="$2"
            shift 2
            ;;
        --dir)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --dir requires a value"
                usage 1
            fi
            DIR_NAME="$2"
            shift 2
            ;;
        --no-gpu)
            REQUIRE_GPU=false
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage 1
            ;;
    esac
done

if [[ -z "$EXP_NUM" && -z "$DIR_NAME" ]]; then
    echo "ERROR: --exp-num or --dir is required"
    usage 1
fi

if [[ -n "$EXP_NUM" && -n "$DIR_NAME" ]]; then
    echo "ERROR: --exp-num and --dir are mutually exclusive"
    usage 1
fi

# Resolve target directory
if [[ -n "$DIR_NAME" ]]; then
    if [[ ! -d "$SCRIPT_DIR/$DIR_NAME" ]]; then
        echo "ERROR: Directory not found: $SCRIPT_DIR/$DIR_NAME"
        exit 1
    fi
    TARGET_DIR_NAME="$DIR_NAME"
else
    EXP_NUM_PADDED=$(printf "%03d" "$EXP_NUM")
    EXP_DIR_MATCH=("$SCRIPT_DIR"/experiments/${EXP_NUM_PADDED}-*/)

    if [[ ! -d "${EXP_DIR_MATCH[0]}" ]]; then
        echo "ERROR: No experiment directory found matching ${EXP_NUM_PADDED}-*"
        echo ""
        echo "Available experiments:"
        for d in "$SCRIPT_DIR"/experiments/[0-9][0-9][0-9]-*/; do
            [ -d "$d" ] && echo "  $(basename "$d")"
        done
        exit 1
    fi

    if [[ ${#EXP_DIR_MATCH[@]} -gt 1 ]]; then
        echo "ERROR: Multiple directories match ${EXP_NUM_PADDED}-*:"
        for d in "${EXP_DIR_MATCH[@]}"; do
            echo "  $(basename "$d")"
        done
        exit 1
    fi

    TARGET_DIR_NAME="experiments/$(basename "${EXP_DIR_MATCH[0]}")"
fi

echo "=========================================="
echo "PlantCLEF 2026 - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] $1${NC}"
    else
        echo -e "${RED}[FAIL] $1${NC}"
        exit 1
    fi
}

# Step 1: Check GPU (unless --no-gpu)
if $REQUIRE_GPU; then
    echo "Step 1: Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        print_status "GPU detected"
    else
        echo -e "${RED}[FAIL] No GPU detected. Use --no-gpu for CPU-only tasks.${NC}"
        exit 1
    fi
else
    echo "Step 1: Skipping GPU check (--no-gpu)"
fi
echo ""

# Step 2: Set up environment variables
echo "Step 2: Setting up environment variables..."
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="$SCRIPT_DIR/$TARGET_DIR_NAME"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo -e "${RED}[FAIL] Target directory not found: $TARGET_DIR${NC}"
    exit 1
fi

# Use TMPDIR if available (SLURM job), otherwise use /tmp
if [ -n "${TMPDIR:-}" ]; then
    VENV_BASE="$TMPDIR"
    echo "  Using SLURM TMPDIR for fast local I/O"
else
    VENV_BASE="/tmp"
    echo "  Using /tmp for interactive session"
fi

JOB_ID="${SLURM_JOB_ID:-$$}"
export JOB_VENV_DIR="$VENV_BASE/plantclef-$JOB_ID"
export JOB_VENV="$JOB_VENV_DIR/.venv"

# Keep uv cache on the same filesystem as the venv to avoid cross-filesystem
# hardlink failures that produce empty stubs.
export UV_CACHE_DIR="$VENV_BASE/uv-cache"

echo "  - Script dir: $SCRIPT_DIR"
echo "  - Project root: $PROJECT_ROOT"
echo "  - Target: $TARGET_DIR_NAME"
echo "  - Target dir: $TARGET_DIR"
echo "  - Job ID: $JOB_ID"
echo "  - Virtual env: $JOB_VENV"
print_status "Environment variables set"
echo ""

# Step 3: Load uv
echo "Step 3: Loading uv for package management..."
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv --version
print_status "uv available"
echo ""

# Step 4: Create virtual environment
echo "Step 4: Setting up venv at $JOB_VENV..."
mkdir -p "$JOB_VENV_DIR"
cd "$TARGET_DIR"
uv venv "$JOB_VENV" --python 3.11
print_status "Virtual environment created"
echo ""

# Step 5: Activate environment
echo "Step 5: Activating virtual environment..."
source "$JOB_VENV/bin/activate"
print_status "Environment activated"
echo ""

# Step 6: Install dependencies
echo "Step 6: Installing dependencies from $TARGET_DIR_NAME/pyproject.toml..."
uv pip install -e "$TARGET_DIR"
print_status "Dependencies installed"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}[OK] ENVIRONMENT SETUP COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Target: $TARGET_DIR_NAME"
echo "Python: $(which python)"
echo "Virtual env: $JOB_VENV"
echo ""
echo "To use this environment:"
echo "  source $JOB_VENV/bin/activate"
echo ""
echo -e "${YELLOW}Note: This environment is temporary and will be cleaned up.${NC}"
echo "SLURM jobs automatically create isolated venvs in \$TMPDIR."
echo ""
