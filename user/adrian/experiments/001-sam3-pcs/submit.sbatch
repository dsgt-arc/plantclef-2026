#!/bin/bash

#SBATCH --job-name=001-sam3-pcs
#SBATCH --output=logs/sam3-%A_%a.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="H100|H200"

#SBATCH --time=04:00:00
#SBATCH --array=0-19   # 20 tasks × ~390 species each = 7,806 species total

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acheung@gatech.edu

set -eu

# ── Paths ────────────────────────────────────────────────────────────────────
scratch_dir=$(realpath ~/scratch/plantclef-2026)
repo_dir=$(realpath ~/arc/plantclef-2026)
exp_dir=$repo_dir/user/adrian/experiments/001-sam3-pcs

# Slice species_list.txt for this task (unequal species sizes are fine —
# the script streams one species at a time regardless of image count)
species_list=$exp_dir/species_list.txt
n_species=$(wc -l < "$species_list")
n_tasks=20
task_id=$SLURM_ARRAY_TASK_ID
start=$(( task_id * n_species / n_tasks + 1 ))   # sed is 1-indexed
end=$(( (task_id + 1) * n_species / n_tasks ))
task_species_file=$TMPDIR/species_task_${task_id}.txt
sed -n "${start},${end}p" "$species_list" > "$task_species_file"

prompt="${prompt:-plant}"
output_dir=$scratch_dir/sam3_outputs/$prompt/task_$(printf "%04d" "$task_id")

# ── Environment ──────────────────────────────────────────────────────────────
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$scratch_dir/.cache/huggingface/hub

module load cuda/13.0.1

# ── GPU sanity check — resubmit if torch can't see a CUDA device ─────────────
if ! uv run --package 001-sam3-pcs python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "WARNING: torch.cuda.is_available()=False on $(hostname) (nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')) — resubmitting task $task_id"
    sbatch --export=ALL,prompt="$prompt" --array="$task_id" "$(realpath "$0")"
    exit 0
fi

echo "=== SAM3 Promptable Concept Segmentation ==="
echo "Array task : $task_id  (species lines ${start}–${end}, $(wc -l < "$task_species_file") species)"
echo "Output     : $output_dir"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Run inference (remote streaming mode) ────────────────────────────────────
mkdir -p "$output_dir"

uv run --package 001-sam3-pcs python src/run_sam3.py \
    --remote      "acheung46@login-phoenix.pace.gatech.edu" \
    --remote_dir  "~/ps-clef2026_plantclef-0/train/PlantCLEF2024/train" \
    --species_ids "$task_species_file" \
    --output_dir  "$output_dir" \
    --stage_dir   "$TMPDIR/sam3_staging" \
    --prompt      "$prompt" \
    --batch_size  32 \
    --num_workers 8 \
    "$@"

echo "Task $task_id complete → $output_dir/results.parquet"
