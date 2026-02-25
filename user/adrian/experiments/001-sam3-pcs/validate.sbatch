#!/bin/bash

#SBATCH --job-name=001-sam3-validate 
#SBATCH --output=logs/validate-%j.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:H100

#SBATCH --time=00:30:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acheung@gatech.edu

set -eu

scratch_dir=$(realpath ~/scratch/plantclef-2026)
repo_dir=$(realpath ~/arc/plantclef-2026)
export exp_dir=$repo_dir/user/adrian/experiments/001-sam3-pcs

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$scratch_dir/.cache/huggingface/hub

module load cuda/13.0.1

echo "=== SAM3 Validation Run ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── Generate random test images ──────────────────────────────────────────────
export test_input=${TMPDIR:-/tmp}/test_chunk
export test_output=${TMPDIR:-/tmp}/test_output

uv run --package 001-sam3-pcs python - <<'EOF'
import os, numpy as np
from pathlib import Path
from PIL import Image

out = Path(os.environ["test_input"])
out.mkdir(parents=True, exist_ok=True)
for i in range(20):
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(arr).save(out / f"test_{i:04d}.jpg")
print(f"Generated {len(list(out.glob('*.jpg')))} test images in {out}")
EOF

# ── Run inference ────────────────────────────────────────────────────────────
uv run --package 001-sam3-pcs python src/run_sam3.py \
    --input_dir  "$test_input" \
    --output_dir "$test_output" \
    --prompt     "plant" \
    --batch_size 4 \
    --num_workers 4

# ── Verify output ────────────────────────────────────────────────────────────
uv run --package 001-sam3-pcs python - <<'EOF'
import os, numpy as np
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.environ["exp_dir"], "src"))
from run_sam3 import rle_decode

out_file = Path(os.environ["test_output"]) / "results.npz"
assert out_file.exists(), f"results.npz not found at {out_file}"

data    = np.load(out_file, allow_pickle=True)
results = data["results"].item()

print(f"\n--- Output summary ---")
print(f"Entries : {len(results)}")
for fname, r in list(results.items())[:3]:
    n = len(r["rle"])
    print(f"  {fname}: {n} mask(s), shape={r['shape'].tolist()}, scores={r['scores'].tolist()}")
    if n > 0:
        mask = rle_decode(r["rle"][0], r["shape"])
        print(f"    RLE round-trip: shape={mask.shape}, dtype={mask.dtype}, coverage={mask.mean():.3f}")

print("\nValidation passed.")
EOF

echo "=== Done ==="
