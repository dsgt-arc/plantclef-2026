"""
Per-species detection rate breakdown for SAM3 inference outputs.

Identifies species where the model consistently fails or succeeds.

Usage:
    UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv uv run --package 001-sam3-pcs \
        python analysis/species_detection_rate.py --prompt plant [--data_dir DIR] [--top_n 20]
"""

import argparse
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(description="SAM3 per-species detection rate")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("~/scratch/plantclef-2026/sam3_outputs").expanduser(),
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt subdirectory name (e.g. 'plant')")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top/bottom species to show")
    parser.add_argument("--top_ids_file", type=Path, default=None, help="Write top_n species IDs to this file (one per line)")
    parser.add_argument("--bottom_ids_file", type=Path, default=None, help="Write bottom_n species IDs to this file (one per line)")
    args = parser.parse_args()

    data_dir: Path = args.data_dir.expanduser() if not args.data_dir.is_absolute() else args.data_dir
    prompt_dir = data_dir / args.prompt

    if not prompt_dir.exists():
        print(f"ERROR: prompt dir not found: {prompt_dir}")
        return

    parquet_files = sorted(prompt_dir.glob("task_*/results.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {prompt_dir}")
        return

    print(f"Loading {len(parquet_files)} parquet files for prompt '{args.prompt}'...")
    df = pl.concat([pl.read_parquet(p) for p in parquet_files])

    # One row per (species_id, filename): was this image detected?
    # mask_idx == -1 means no detection for that image
    per_image = (
        df.group_by(["species_id", "filename"])
        .agg(
            pl.col("mask_idx").max().alias("max_mask_idx")
        )
        .with_columns(
            (pl.col("max_mask_idx") >= 0).alias("detected")
        )
    )

    species_stats = (
        per_image.group_by("species_id")
        .agg(
            pl.len().alias("total_images"),
            pl.col("detected").sum().alias("detected_images"),
        )
        .with_columns(
            (pl.col("detected_images") / pl.col("total_images") * 100.0).alias("det_rate_pct")
        )
        .sort("det_rate_pct", descending=True)
    )

    total_species = len(species_stats)
    zero_det = species_stats.filter(pl.col("det_rate_pct") == 0.0)
    total_images = per_image.height

    print(f"\nTotal species: {total_species:,}")
    print(f"Total images:  {total_images:,}")
    print(f"Species with 0% detection rate: {len(zero_det):,}")

    def print_species_table(title: str, rows: pl.DataFrame) -> None:
        print(f"\n{title}")
        header = f"{'species_id':<20} {'images':>8} {'detected':>10} {'det%':>8}"
        sep = "-" * len(header)
        print(header)
        print(sep)
        for row in rows.iter_rows(named=True):
            print(
                f"{str(row['species_id']):<20} {row['total_images']:>8,} "
                f"{row['detected_images']:>10,} {row['det_rate_pct']:>7.1f}%"
            )
        print(sep)

    top_n = min(args.top_n, total_species)
    top_species    = species_stats.head(top_n)
    bottom_species = species_stats.tail(top_n).sort("det_rate_pct")

    print_species_table(f"Top {top_n} species by detection rate:", top_species)
    print_species_table(f"Bottom {top_n} species by detection rate:", bottom_species)

    if args.top_ids_file:
        args.top_ids_file.write_text("\n".join(top_species["species_id"].to_list()) + "\n")
        print(f"\nTop IDs written to: {args.top_ids_file}")
    if args.bottom_ids_file:
        args.bottom_ids_file.write_text("\n".join(bottom_species["species_id"].to_list()) + "\n")
        print(f"Bottom IDs written to: {args.bottom_ids_file}")


if __name__ == "__main__":
    main()
