#!/bin/bash
#SBATCH --job-name=exp14_stage
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --qos=embers
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu
#SBATCH --output=logs/exp14_stage_%j.out

# Run one stage's variant sweep on PACE. CPU-only — no GPU needed for
# fusion + ensemble + CSV generation.
#
# Required env:
#   STAGE_SPEC       — path to a stage_configs/*.json file (relative to Exp-14 dir)
#   VIT_PREFUSION    — pre-fusion ViT parquet
#   CNXT_SCALE_4     — scale 4x4 ConvNeXt parquet
#   CNXT_SCALE_5     — scale 5x5
#   CNXT_SCALE_6     — scale 6x6
#   SPECIES_MAP      — species map JSON
#   CLASS_MASK       — geo class mask npy
#   BASELINE_CSV     — path to ens_knn_w0.77_thr0.100_top10.csv (for delta proxy)

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
EXP14_DIR="$(pwd)"
OUT_DIR="${EXP14_DIR}/outputs/$(basename "$STAGE_SPEC" .json)"
mkdir -p "$OUT_DIR" logs

DENSITY_OPT=""
if [[ -f "${EXP14_DIR}/outputs/signals/density.parquet" ]]; then
    DENSITY_OPT="--density ${EXP14_DIR}/outputs/signals/density.parquet"
fi
SIM_OPT=""
if [[ -f "${EXP14_DIR}/outputs/signals/visit_similarity.json" ]]; then
    SIM_OPT="--sim-json ${EXP14_DIR}/outputs/signals/visit_similarity.json"
fi
SEASON_OPT=""
if [[ -f "${EXP14_DIR}/outputs/signals/season_weights.json" ]]; then
    SEASON_OPT="--season-json ${EXP14_DIR}/outputs/signals/season_weights.json"
fi

echo "=== Exp-14 stage sweep ==="
echo "  spec:    $STAGE_SPEC"
echo "  out:     $OUT_DIR"

uv run python scripts/stage_runner.py \
    --stage-spec "$STAGE_SPEC" \
    --out-dir "$OUT_DIR" \
    --vit-prefusion-parquet "$VIT_PREFUSION" \
    --convnext-scales "$CNXT_SCALE_4" "$CNXT_SCALE_5" "$CNXT_SCALE_6" \
    --species-map "$SPECIES_MAP" \
    --class-mask-npy "$CLASS_MASK" \
    --temperature 1.5 \
    --weights 0.77 \
    --thresholds 0.100 \
    --topk 10 \
    --baseline-csv "$BASELINE_CSV" \
    $DENSITY_OPT $SIM_OPT $SEASON_OPT

echo "=== Done ==="
