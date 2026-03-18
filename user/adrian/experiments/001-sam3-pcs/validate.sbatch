#!/bin/bash

#SBATCH --job-name=001-sam3-validate
#SBATCH --output=logs/validate-%j.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="H100|H200"

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
export remote_dir='~/ps-clef2026_plantclef-0/train/PlantCLEF2024/train'

module load cuda/13.0.1

echo "=== SAM3 Validation Run ==="
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

prompt="${prompt:-plant}"
export output_dir=$TMPDIR/sam3_validate_output_$prompt

# ── Run inference on 5 real species ─────────────────────────────────────────
uv run --package 001-sam3-pcs python src/run_sam3.py \
    --remote      "acheung46@login-phoenix.pace.gatech.edu" \
    --remote_dir  "$remote_dir" \
    --species_ids "$exp_dir/species_test.txt" \
    --output_dir  "$output_dir" \
    --stage_dir   "$TMPDIR/sam3_staging" \
    --prompt      "$prompt" \
    --batch_size  32 \
    --num_workers 8

# ── Report ───────────────────────────────────────────────────────────────────
uv run --package 001-sam3-pcs python - <<'EOF'
import os, sys
import numpy as np
import polars as pl
from pathlib import Path
sys.path.insert(0, os.path.join(os.environ["exp_dir"], "src"))
from run_sam3 import rle_decode

out_file = Path(os.environ["output_dir"]) / "results.parquet"
assert out_file.exists(), f"results.parquet not found at {out_file}"

df = pl.read_parquet(out_file)

print(f"\n--- Output summary ---")
print(f"Total rows     : {len(df):,}")
print(f"Total images   : {df['filename'].n_unique():,}")
print(f"Species seen   : {df['species_id'].n_unique()}")
print(f"Detections     : {(df['mask_idx'] >= 0).sum():,}")
print(f"No-detection   : {(df['mask_idx'] < 0).sum():,}")
print(f"Score stats    :\n{df.filter(pl.col('mask_idx') >= 0)['score'].describe()}")
print(f"\nFile size      : {out_file.stat().st_size / 1e6:.2f} MB")

# RLE round-trip check on first detection
first = df.filter(pl.col("mask_idx") == 0).head(1)
if len(first) > 0:
    row   = first.row(0, named=True)
    rle   = np.frombuffer(row["rle"], dtype=np.int32)
    shape = (row["shape_h"], row["shape_w"])
    mask  = rle_decode(rle, shape)
    print(f"\nRLE round-trip : shape={mask.shape}, coverage={mask.mean():.4f}")

print(f"\nSample rows:")
print(df.head(5))
print("\nValidation passed.")
EOF

# ── Visualizations ───────────────────────────────────────────────────────────
# Pick 2 images per species that have at least one detection, rsync them back,
# draw semi-transparent mask overlays + bounding boxes, save to scratch so you
# can scp/view them after the job.
export vis_dir=$scratch_dir/sam3_validate_vis
mkdir -p "$vis_dir"

uv run --package 001-sam3-pcs python - <<'EOF'
import os, sys, subprocess
import numpy as np
import polars as pl
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.environ["exp_dir"], "src"))
from run_sam3 import rle_decode

REMOTE        = "acheung46@login-phoenix.pace.gatech.edu"
REMOTE_DIR    = os.environ["remote_dir"]
output_dir    = Path(os.environ["output_dir"])
vis_dir       = Path(os.environ["vis_dir"])
stage_dir     = Path(os.environ["TMPDIR"]) / "vis_staging"
stage_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(output_dir / "results.parquet")

# Pick up to 2 images per species that have detections
samples = (
    df.filter(pl.col("mask_idx") == 0)          # one row per image with detections
      .group_by("species_id")
      .agg(pl.col("filename").head(2))
      .explode("filename")
)

# Colour palette for up to 10 masks per image (RGBA)
COLOURS = [
    (255,  80,  80, 120),
    ( 80, 200,  80, 120),
    ( 80, 120, 255, 120),
    (255, 200,  50, 120),
    (200,  80, 255, 120),
    ( 50, 220, 220, 120),
    (255, 140,   0, 120),
    (  0, 200, 150, 120),
    (220,  50,  50, 120),
    (100, 100, 255, 120),
]

n_saved = 0
for row in samples.iter_rows(named=True):
    species_id = row["species_id"]
    filename   = row["filename"]

    # Rsync the single image
    local_img = stage_dir / filename
    src = f"{REMOTE}:{REMOTE_DIR}/{species_id}/{filename}"
    result = subprocess.run(
        ["rsync", "-aq", "--no-perms", src, str(local_img)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  rsync failed for {filename}: {result.stderr.strip()}")
        continue

    # Load image
    img  = Image.open(local_img).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Get all masks for this image
    img_masks = df.filter(
        (pl.col("species_id") == species_id) &
        (pl.col("filename")   == filename)   &
        (pl.col("mask_idx")   >= 0)
    ).sort("mask_idx")

    for i, mask_row in enumerate(img_masks.iter_rows(named=True)):
        rle   = np.frombuffer(mask_row["rle"], dtype=np.int32)
        shape = (mask_row["shape_h"], mask_row["shape_w"])
        mask  = rle_decode(rle, shape)

        # Semi-transparent colour fill
        colour = COLOURS[i % len(COLOURS)]
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        overlay_arr[mask] = colour
        img = Image.alpha_composite(img, Image.fromarray(overlay_arr, "RGBA"))

        # Bounding box + score label
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = mask_row["box_x1"], mask_row["box_y1"], mask_row["box_x2"], mask_row["box_y2"]
        box_colour = colour[:3] + (220,)
        draw.rectangle([x1, y1, x2, y2], outline=box_colour, width=2)
        draw.text((x1 + 3, y1 + 3), f"{mask_row['score']:.2f}", fill=(255, 255, 255, 230))

    out_path = vis_dir / f"{species_id}_{Path(filename).stem}.png"
    img.convert("RGB").save(out_path)
    print(f"  Saved {out_path.name}  ({len(img_masks)} mask(s))")
    n_saved += 1
    local_img.unlink(missing_ok=True)

print(f"\nSaved {n_saved} visualizations → {vis_dir}")
EOF

echo "=== Done ==="
echo "Visualizations at: $vis_dir"
echo "Copy with: scp -r $(hostname):$vis_dir ~/Downloads/sam3_vis"
