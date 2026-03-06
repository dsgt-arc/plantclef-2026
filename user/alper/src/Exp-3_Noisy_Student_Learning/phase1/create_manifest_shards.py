#!/usr/bin/env python3
"""Create shard manifest files for SLURM array parallelization.

Two modes:
  --mode lucas : Read 77k metadata CSV, map image_name to full paths in 212k dir
  --mode test  : Scan test images dir directly
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_lucas_paths(
    metadata_csv: str,
    image_dir: str,
    image_col: str = "image_name",
) -> list[Path]:
    df = pd.read_csv(metadata_csv)
    if image_col not in df.columns:
        raise ValueError(
            f"Column '{image_col}' not found in {metadata_csv}. "
            f"Available: {list(df.columns)}"
        )
    image_names = df[image_col].dropna().unique()
    img_dir = Path(image_dir)
    paths = []
    missing = 0
    for name in sorted(image_names):
        p = img_dir / str(name)
        if p.exists():
            paths.append(p)
        else:
            missing += 1
    print(f"Found {len(paths)} images, {missing} missing from {img_dir}")
    return paths


def collect_test_paths(image_dir: str) -> list[Path]:
    img_dir = Path(image_dir)
    paths = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Found {len(paths)} test images in {img_dir}")
    return paths


def write_shards(
    paths: list[Path],
    output_dir: str,
    shard_size: int = 1000,
    prefix: str = "shard",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write master manifest
    all_path = out / "all.txt"
    with open(all_path, "w") as f:
        for p in paths:
            f.write(str(p) + "\n")
    print(f"Wrote master manifest: {all_path} ({len(paths)} entries)")

    # Write shard manifests
    num_shards = math.ceil(len(paths) / shard_size)
    for shard_id in range(num_shards):
        shard_path = out / f"{prefix}_{shard_id:04d}.txt"
        start = shard_id * shard_size
        end = min(start + shard_size, len(paths))
        with open(shard_path, "w") as f:
            for p in paths[start:end]:
                f.write(str(p) + "\n")

    print(f"Created {num_shards} shards (size={shard_size})")
    if num_shards > 0:
        last_size = len(paths) - (num_shards - 1) * shard_size
        print(f"Last shard size: {last_size}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["lucas", "test"],
        help="'lucas' reads metadata CSV; 'test' scans image dir",
    )
    ap.add_argument("--metadata-csv", default=None, help="Path to 77k metadata CSV (lucas mode)")
    ap.add_argument("--image-dir", default=None, help="Directory containing images")
    ap.add_argument("--image-col", default="image_name", help="CSV column with image filenames")
    ap.add_argument("--output-dir", required=True, help="Where to write shard manifests")
    ap.add_argument("--shard-size", type=int, default=1000, help="Images per shard")

    args = ap.parse_args()

    if args.mode == "lucas":
        if not args.metadata_csv:
            ap.error("--metadata-csv is required for lucas mode")
        if not args.image_dir:
            ap.error("--image-dir is required for lucas mode")
        paths = collect_lucas_paths(args.metadata_csv, args.image_dir, args.image_col)
    else:
        if not args.image_dir:
            ap.error("--image-dir is required for test mode")
        paths = collect_test_paths(args.image_dir)

    if not paths:
        raise RuntimeError("No images found. Check paths and arguments.")

    write_shards(paths, args.output_dir, args.shard_size)


if __name__ == "__main__":
    main()
