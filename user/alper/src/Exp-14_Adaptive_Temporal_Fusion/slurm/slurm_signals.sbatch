#!/bin/bash
#SBATCH --job-name=exp14_signals
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --qos=embers
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu
#SBATCH --output=logs/exp14_signals_%j.out

# Compute visit similarity from existing per-tile DINOv2 embeddings.
# Run BEFORE any sim_w aggregator stage. Output is a JSON ~50-100 MB.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
EXP14_DIR="$(pwd)"
TILE_DIR="${PLANTCLEF_ROOT:-/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026}/parquet/exp9b_knn/embeddings_arcface_lora/test"
OUT_JSON="${EXP14_DIR}/outputs/signals/visit_similarity.json"

mkdir -p "$(dirname "$OUT_JSON")" logs

# Use frozen DINOv2 embedding (embedding_orig_f16) — better for "is this lush
# vs barren" than ArcFace which is species-decision-aligned.
PARQUETS=$(ls "${TILE_DIR}"/scale_*.parquet)

echo "=== Exp-14 visit similarity ==="
echo "  Parquets: $PARQUETS"
echo "  Output:   $OUT_JSON"

uv run python signals/compute_visit_similarity.py \
    --parquets $PARQUETS \
    --emb-col embedding_orig_f16 \
    --out "$OUT_JSON"

echo "=== Done ==="
