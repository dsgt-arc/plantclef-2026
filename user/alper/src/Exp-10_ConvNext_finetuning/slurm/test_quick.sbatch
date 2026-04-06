#!/bin/bash
#SBATCH --job-name=cnxt-test
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_%x_%A.out
#SBATCH --error=logs/test_%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext_test"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/logs"

echo "=== Quick Validation Test ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import timm; print(f'timm {timm.__version__}')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')"

# ---- Test 1: Verify data directory exists and has expected structure ----
echo ""
echo "=== Test 1: Data directory ==="
echo "Data dir: ${DATA_DIR}"
ls -d "${DATA_DIR}/train" "${DATA_DIR}/val" 2>&1
echo "Train classes: $(ls "${DATA_DIR}/train" | wc -l)"
echo "Val classes: $(ls "${DATA_DIR}/val" | wc -l)"
echo "Sample train images: $(ls "${DATA_DIR}/train/1355868" | head -3)"

# ---- Test 2: Verify model loading with pretrained weights ----
echo ""
echo "=== Test 2: Model loading ==="
python "${EXP_DIR}/scripts/verify_model_loading.py" --pretrained

# ---- Test 3: Phase 1 training (1 epoch, small batch, frozen backbone) ----
echo ""
echo "=== Test 3: Phase 1 training (1 epoch) ==="
python "${EXP_DIR}/train_convnext.py" \
    --model convnext_tiny.dinov3_lvd1689m \
    --pretrained \
    --num-classes 7806 \
    --freeze-backbone \
    --input-size 3 224 224 \
    --epochs 1 \
    --batch-size 32 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.0 \
    --sched cosine \
    --smoothing 0.1 \
    --amp \
    --amp-dtype float16 \
    --channels-last \
    --model-ema \
    --model-ema-decay 0.9998 \
    --workers 4 \
    --pin-mem \
    --seed 42 \
    --log-interval 50 \
    --eval-metric top1 \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    --output "${OUTPUT_DIR}" \
    --experiment test_phase1

# ---- Test 4: Verify outputs ----
echo ""
echo "=== Test 4: Verify outputs ==="
PHASE1_DIR="${OUTPUT_DIR}/test_phase1"
echo "Output directory contents:"
ls -la "${PHASE1_DIR}/"

echo ""
echo "summary.csv:"
cat "${PHASE1_DIR}/summary.csv"

echo ""
echo "args.yaml (first 20 lines):"
head -20 "${PHASE1_DIR}/args.yaml"

echo ""
echo "Checkpoint exists:"
ls -lh "${PHASE1_DIR}/model_best.pth.tar" "${PHASE1_DIR}/last.pth.tar" 2>&1

# ---- Test 5: Phase 2 dry run (1 epoch, loading Phase 1 checkpoint) ----
echo ""
echo "=== Test 5: Phase 2 training (1 epoch, from Phase 1 checkpoint) ==="
python "${EXP_DIR}/train_convnext.py" \
    --model convnext_tiny.dinov3_lvd1689m \
    --num-classes 7806 \
    --initial-checkpoint "${PHASE1_DIR}/model_best.pth.tar" \
    --input-size 3 224 224 \
    --epochs 1 \
    --batch-size 32 \
    --opt adamw \
    --lr 8e-5 \
    --weight-decay 0.05 \
    --layer-decay 0.7 \
    --sched cosine \
    --warmup-lr 1e-6 \
    --warmup-epochs 1 \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --amp \
    --amp-dtype float16 \
    --channels-last \
    --model-ema \
    --model-ema-decay 0.9998 \
    --workers 4 \
    --pin-mem \
    --seed 42 \
    --log-interval 50 \
    --eval-metric top1 \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    --output "${OUTPUT_DIR}" \
    --experiment test_phase2

echo ""
echo "=== Test 5: Phase 2 outputs ==="
PHASE2_DIR="${OUTPUT_DIR}/test_phase2"
echo "summary.csv:"
cat "${PHASE2_DIR}/summary.csv"

echo ""
echo "=== ALL TESTS PASSED ==="
