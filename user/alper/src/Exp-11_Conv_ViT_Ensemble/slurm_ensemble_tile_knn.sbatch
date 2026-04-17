#!/bin/bash
# Exp-11: Tile-level ensemble + kNN
# Blend ViT+ConvNeXt logits per tile → kNN on ViT embeddings → aggregate → temporal fusion

#SBATCH --job-name=ens-tile-knn
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ens_tile_knn_%A.out
#SBATCH --error=logs/ens_tile_knn_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP11_DIR="${REPO_ROOT}/user/alper/src/Exp-11_Conv_ViT_Ensemble"
EXP9_DIR="${REPO_ROOT}/user/alper/src/Exp-9_kNN_Retrieval_Ensemble"
EXP8_DIR="${REPO_ROOT}/user/alper/src/Exp-8_Quantile_Aggregation_SS"

export PYTHONPATH="${EXP9_DIR}:${EXP8_DIR}:${PYTHONPATH:-}"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

# ViT tiles
VIT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"
VIT_4x4="${VIT_DIR}/scale_4x4.parquet"

# ConvNeXt tiles
CNXT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large"

# kNN index
FAISS_INDEX="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_index_geo.bin"
FAISS_LABELS="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/faiss_labels_geo.npy"

# Mapping
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
CLASS_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/exp11_tile_knn"

mkdir -p "${EXP11_DIR}/logs"
mkdir -p "${OUT_DIR}"

echo "=== Tile-level Ensemble + kNN ==="
echo "Date: $(date)"

python "${EXP11_DIR}/ensemble_tile_knn.py" \
    --vit-scales \
        "${VIT_4x4}" \
        "${VIT_DIR}/scale_5x5.parquet" \
        "${VIT_DIR}/scale_6x6.parquet" \
    --cnxt-scales \
        "${CNXT_DIR}/scale_4x4.parquet" \
        "${CNXT_DIR}/scale_5x5.parquet" \
        "${CNXT_DIR}/scale_6x6.parquet" \
    --faiss-index "${FAISS_INDEX}" \
    --faiss-labels "${FAISS_LABELS}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${CLASS_MASK}" \
    --out-dir "${OUT_DIR}" \
    --temperature 1.5 \
    --alpha 0.15 \
    --knn-k 20 \
    --knn-tau 0.07 \
    --knn-beta-multis "0.70,0.80,0.90" \
    --knn-beta-single 1.00 \
    --blend-weights "0.78,0.80,0.85,0.90,1.00" \
    --thresholds "0.100" \
    --topk 10

echo "=== Done ==="
