#!/bin/bash
#SBATCH --job-name=cnxt-aggG-coollast
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PLANTCLEF_ROOT=/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0
REPO_ROOT=/storage/home/hcoda1/2/aerten3/plantclef-2026
SRC_ROOT=$REPO_ROOT/user/alper/src/Exp-11C_ViT-L_KNN_ArcFace_Conv_Ensemble

mkdir -p logs
source $REPO_ROOT/user/alper/experiments/001-embedding-extraction/.venv/bin/activate

OUT=$PLANTCLEF_ROOT/aerten3/submissions/exp10c_diag_scheme_G
mkdir -p $OUT

python $SRC_ROOT/multiscale_aggregate.py \
    --scale5-dir $PLANTCLEF_ROOT/aerten3/parquet/exp11_ensemble/convnext_large_phase2v3/scale_5x5.parquet \
    --model-path $PLANTCLEF_ROOT/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth \
    --species-map $PLANTCLEF_ROOT/data/raw/class_id_to_species_mapping/class_to_species_id.json \
    --class-mask-npy $PLANTCLEF_ROOT/aerten3/species_masking/sw_europe_mask_bool.npy \
    --out-dir $OUT --temperature 1.5 --alpha 0.30 --topk 10 --image-topk 64 \
    --schemes "G" --thresholds "0.085,0.100,0.125" --csv-prefix "cooldown_last_"
