#!/bin/bash
# topk_sweep.sh — Regenerate submission CSVs with different topk values.
# CPU-only, takes seconds per run.
#
# Usage:
#   bash topk_sweep.sh <aggregated-parquet> [out-dir] [probs-col]
#
# Example:
#   bash topk_sweep.sh /path/to/sweep_T2.0_k6.parquet                          # Exp-3 parquet (auto-detects column)
#   bash topk_sweep.sh /path/to/mv_avg_logits_noisy_or_T2.0_k6.parquet /out/   # Exp-4 parquet
set -euo pipefail

PARQUET_PATH="${1:?Usage: $0 <aggregated-parquet> [out-dir] [probs-col]}"
OUT_DIR="${2:-$(dirname "${PARQUET_PATH}")/topk_sweep}"
PROBS_COL="${3:-auto}"

# Auto-detect probs column: Exp-4 uses "image_probs_f16", Exp-3 uses "teacher_image_probs_f16"
if [ "${PROBS_COL}" = "auto" ]; then
  PROBS_COL=$(python -c "
import pyarrow.parquet as pq
schema = pq.read_schema('${PARQUET_PATH}')
print('image_probs_f16' if 'image_probs_f16' in schema.names else 'teacher_image_probs_f16')
" 2>/dev/null) || PROBS_COL="image_probs_f16"
  echo "Auto-detected probs column: ${PROBS_COL}"
fi

PLANTCLEF_ROOT="${PLANTCLEF_ROOT:-/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0}"
SRC_ROOT="${SRC_ROOT:-/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-4_Enhanced_Tile_Aggregation}"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

mkdir -p "${OUT_DIR}"

BASENAME=$(basename "${PARQUET_PATH}" .parquet)

echo "=== topk sweep for ${BASENAME} ==="

for K in 5 8 10 12 15 20; do
  for THR in 0.03 0.0; do
    CSV="${OUT_DIR}/${BASENAME}_top${K}_thr${THR}.csv"
    echo "  topk=${K} threshold=${THR} -> ${CSV}"
    python "${SRC_ROOT}/generate_submission.py" \
      --in-parquet "${PARQUET_PATH}" \
      --out-csv "${CSV}" \
      --id-col "image_name" \
      --probs-col "${PROBS_COL}" \
      --species-map "${SPECIES_MAP}" \
      --class-mask-npy "${GEO_MASK}" \
      --topk "${K}" \
      --threshold "${THR}" \
      --min-k 1
  done
done

echo ""
echo "=== topk sweep complete ==="
echo "CSVs in: ${OUT_DIR}/"
