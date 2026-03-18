#!/usr/bin/env python3
"""Convert existing image-level parquet files to per-tile format.

Reads parquets with schema (image_name, grid_size, tiles_per_image, probs)
and rewrites them with schema (image_name, tile_idx, probs) where each row
is a single tile's probability vector.

Usage:
    # Convert all parquets in the inference directory (overwrites in place)
    python convert_parquet.py

    # Convert a specific file
    python convert_parquet.py --input /path/to/grid_4x4.parquet

    # Dry run (print stats without writing)
    python convert_parquet.py --dry-run
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DIR = "/storage/project/ps-clef2026_plantclef-0/shared/inference/dinov2-vit-base"
NUM_CLASSES = 7806


def is_old_format(path: Path) -> bool:
    """Check if parquet uses the old image-level format."""
    schema = pq.read_schema(path)
    return "tiles_per_image" in schema.names


def convert_file(path: Path, dry_run: bool = False) -> None:
    df = pq.read_table(path).to_pandas()

    if "tiles_per_image" not in df.columns:
        print(f"  SKIP {path.name} (already per-tile format)")
        return

    old_rows = len(df)
    rows = {"image_name": [], "tile_idx": [], "probs": []}

    for _, row in df.iterrows():
        tiles = row["tiles_per_image"]
        probs = np.frombuffer(row["probs"], dtype=np.float16).reshape(tiles, NUM_CLASSES)
        for t in range(tiles):
            rows["image_name"].append(row["image_name"])
            rows["tile_idx"].append(t)
            rows["probs"].append(probs[t].tobytes())

    new_rows = len(rows["image_name"])
    print(f"  {path.name}: {old_rows} images -> {new_rows} tiles")

    if dry_run:
        return

    table = pa.table(
        {
            "image_name": pa.array(rows["image_name"], type=pa.string()),
            "tile_idx": pa.array(rows["tile_idx"], type=pa.int32()),
            "probs": pa.array(rows["probs"], type=pa.binary()),
        }
    )
    pq.write_table(table, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Written: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert parquets to per-tile format")
    parser.add_argument("--input", type=str, help="Single parquet file to convert")
    parser.add_argument("--dir", type=str, default=DEFAULT_DIR, help="Directory of parquets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.input:
        files = [Path(args.input)]
    else:
        files = sorted(Path(args.dir).glob("*.parquet"))

    print(f"Found {len(files)} parquet file(s)")
    for f in files:
        convert_file(f, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
