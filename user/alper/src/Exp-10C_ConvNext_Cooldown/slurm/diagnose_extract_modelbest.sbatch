#!/bin/bash
# Exp-10C diagnostic — extract scale 5 with model_best.pth.tar instead of last.pth.tar.
#
# Cooldown last.pth.tar standalone scored 0.27 vs production 0.36 at scheme P,
# thr=0.085 — −0.087 F1 regression. ViT-L cooldown gained +0.011 at the same
# recipe family, so the bug is ConvNeXt-specific. Test whether last.pth.tar
# is the problem (final-batch artifact) or training itself failed.
#
# Scale 5 chosen for speed (25 tiles/img vs 36 at scale 6 — ~1.4× faster).
#
# Output: ${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_large_phase2v3_modelbest/
#   scale_5x5.parquet
#
# After this finishes, aggregate with multiscale_aggregate.py and compare
# scheme G (scale 5 alone) F1 to:
#   - production scale 5: aggregate same scheme G from production parquets
#   - cooldown_last scale 5: aggregate same scheme G from cooldown_last parquets
#
# Usage:
#   sbatch slurm/diagnose_extract_modelbest.sbatch

#SBATCH --job-name=cnxt-diag-modelbest
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --qos=inferno
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/diag_%x_%j.out
#SBATCH --error=logs/diag_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

VARIANT="${VARIANT:-large}"
SCALE="${SCALE:-5}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"

IMAGE_DIR="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/parquet/exp11_ensemble/convnext_${VARIANT}_phase2v3_modelbest"

# Use model_best.pth.tar — even though val=train_full and val metric is
# "contaminated", model_best still encapsulates a valid intermediate state
CKPT="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext/convnext_${VARIANT}_phase2v3/model_best.pth.tar"

case "${VARIANT}" in
    tiny)  MODEL_NAME="convnext_tiny.dinov3_lvd1689m" ;;
    small) MODEL_NAME="convnext_small.dinov3_lvd1689m" ;;
    base)  MODEL_NAME="convnext_base.dinov3_lvd1689m" ;;
    large) MODEL_NAME="convnext_large.dinov3_lvd1689m" ;;
    *)     echo "Unknown variant: ${VARIANT}"; exit 1 ;;
esac

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: model_best checkpoint not found at ${CKPT}"
    echo "List ckpt dir contents:"
    ls -lh "$(dirname ${CKPT})"
    exit 1
fi

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs "${OUTPUT_DIR}"

echo "=== DIAGNOSTIC: extract scale ${SCALE} with model_best.pth.tar ==="
echo "Checkpoint: ${CKPT}"
ls -lh "${CKPT}"

echo ""
echo "--- Step 1: Inspect checkpoint contents ---"
python -c "
import torch
ckpt = torch.load('${CKPT}', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
print('Epoch:', ckpt.get('epoch'))
print('Metric:', ckpt.get('metric'))
print('Arch:', ckpt.get('arch'))
sd_keys = list(ckpt.get('state_dict', {}).keys())[:5]
sde_keys = list(ckpt.get('state_dict_ema', {}).keys())[:5]
print('state_dict first 5 keys:', sd_keys)
print('state_dict_ema first 5 keys:', sde_keys)
print('state_dict size:', sum(v.numel() for v in ckpt.get('state_dict', {}).values()))
print('state_dict_ema size:', sum(v.numel() for v in ckpt.get('state_dict_ema', {}).values()))
# Compare regular vs EMA
sd = ckpt.get('state_dict', {})
sde = ckpt.get('state_dict_ema', {})
# Strip 'module.' prefix if present
def strip(d):
    return {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in d.items()}
sd, sde = strip(sd), strip(sde)
common = set(sd.keys()) & set(sde.keys())
if common:
    diffs = []
    for k in list(common)[:50]:
        if sd[k].dtype.is_floating_point:
            diffs.append((sd[k].float() - sde[k].float()).abs().mean().item())
    print(f'Mean abs diff (regular vs EMA, sampled): {sum(diffs)/len(diffs):.6f}' if diffs else 'no float params sampled')
"

echo ""
echo "--- Step 2: Extract scale ${SCALE} with model_best ---"
python "${EXP_DIR}/extract_convnext_embeddings.py" \
    --scale "${SCALE}" \
    --image-dir "${IMAGE_DIR}" \
    --model-path "${CKPT}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --jpeg-quality 85 \
    --batch-size 64

echo ""
echo "=== Done. Output: ${OUTPUT_DIR}/scale_${SCALE}x${SCALE}.parquet ==="
echo ""
echo "Next: aggregate at scheme G (scale 5 alone) and compare F1 vs production/cooldown_last."
