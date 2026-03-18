#!/bin/bash

#SBATCH --job-name=001-sam3-vis
#SBATCH --output=logs/vis-%j.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:30:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acheung@gatech.edu

# Usage examples:
#
#   # Random 10 species, 5 images each (all prompts)
#   sbatch visualize.sbatch
#
#   # Specific species
#   SPECIES="1356390 1743350" sbatch visualize.sbatch
#
#   # 20 species, plant prompt only, 3 images max, reproducible seed
#   N_SPECIES=20 PROMPT=plant MAX_IMAGES=3 SEED=42 sbatch visualize.sbatch

set -eu

scratch_dir=$(realpath ~/scratch/plantclef-2026)
repo_dir=$(realpath ~/arc/plantclef-2026)
exp_dir=$repo_dir/user/adrian/experiments/001-sam3-pcs

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$scratch_dir/.cache/huggingface/hub

echo "=== SAM3 Mask Visualization ==="
echo "Node : $(hostname)"
echo "Date : $(date)"

# Build CLI args from environment variables
extra_args=()

if [[ -n "${SPECIES:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args+=(--species $SPECIES)
elif [[ -n "${N_SPECIES:-}" ]]; then
    extra_args+=(--n_species "$N_SPECIES")
else
    extra_args+=(--n_species 10)
fi

if [[ -n "${MAX_IMAGES:-}" ]]; then
    extra_args+=(--max_images "$MAX_IMAGES")
fi

if [[ -n "${SEED:-}" ]]; then
    extra_args+=(--seed "$SEED")
fi

if [[ -n "${PROMPT:-}" ]]; then
    extra_args+=(--prompt "$PROMPT")
fi

if [[ -n "${SCORE_THRESHOLD:-}" ]]; then
    extra_args+=(--score_threshold "$SCORE_THRESHOLD")
fi

if [[ -n "${WORKERS:-}" ]]; then
    extra_args+=(--workers "$WORKERS")
fi

if [[ -n "${OUT_DIR:-}" ]]; then
    extra_args+=(--out_dir "$OUT_DIR")
fi

echo "Args : ${extra_args[*]}"

cd "$exp_dir"
uv run --package 001-sam3-pcs python analysis/visualize_masks.py \
    --stage_dir "$TMPDIR/vis_staging" \
    "${extra_args[@]}"

echo "=== Done ==="
echo "Output: $scratch_dir/sam3_vis/"
echo "Copy with: scp -r $(hostname):$scratch_dir/sam3_vis ~/Downloads/sam3_vis"
