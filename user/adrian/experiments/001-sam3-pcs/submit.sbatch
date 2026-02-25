#!/bin/bash

#SBATCH --job-name=001-sam3-pcs
#SBATCH --output=logs/sam3-%A_%a.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:H100

#SBATCH --time=06:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acheung@gatech.edu

set -eu

# ── Paths ────────────────────────────────────────────────────────────────────
scratch_dir=$(realpath ~/scratch/plantclef-2026)
repo_dir=$(realpath ~/arc/plantclef-2026)
exp_dir=$repo_dir/user/adrian/experiments/001-sam3-pcs

chunk_id=$(printf "%06d" "$SLURM_ARRAY_TASK_ID")
input_scratch=$scratch_dir/chunks/chunk_$chunk_id
output_dir=$scratch_dir/sam3_outputs/chunk_$chunk_id

# ── Environment ──────────────────────────────────────────────────────────────
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$scratch_dir/.cache/huggingface/hub   # cache model weights on scratch

module load cuda/12.6.0

echo "=== SAM3 Promptable Concept Segmentation ==="
echo "Array task : $SLURM_ARRAY_TASK_ID  (chunk $chunk_id)"
echo "Input      : $input_scratch"
echo "Output     : $output_dir"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Stage input chunk to local NVMe ─────────────────────────────────────────
# Copying to $TMPDIR avoids repeated hits on the network scratch during the
# DataLoader's random-access image reads.
stage_dir=$TMPDIR/chunk_$chunk_id
echo "Staging images to $stage_dir ..."
cp -r "$input_scratch" "$stage_dir"
echo "Staged $(find "$stage_dir" -maxdepth 1 -name '*.jpg' -o -name '*.jpeg' | wc -l) images."

# ── Run inference ────────────────────────────────────────────────────────────
mkdir -p "$output_dir"

uv run --project "$exp_dir" python src/run_sam3.py \
    --input_dir   "$stage_dir" \
    --output_dir  "$output_dir" \
    --prompt      "plant" \
    --batch_size  8 \
    --num_workers 8 \
    "$@"

echo "Chunk $chunk_id complete. Output: $output_dir/results.npz"
