"""
Prompt agreement analysis: compare plant vs flower SAM3 detections per image.

Computes a 2x2 contingency table and per-species disagreement breakdown.

Usage:
    UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv uv run --package 001-sam3-pcs \
        python analysis/prompt_agreement.py [--data_dir DIR] [--top_n 20]
"""

import argparse
from pathlib import Path

import polars as pl


def load_detection_map(prompt_dir: Path) -> pl.DataFrame:
    """Load all parquets and return (species_id, filename, detected) per image."""
    parquet_files = sorted(prompt_dir.glob("task_*/results.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {prompt_dir}")

    df = pl.concat([pl.read_parquet(p) for p in parquet_files])

    return (
        df.group_by(["species_id", "filename"])
        .agg(pl.col("mask_idx").max().alias("max_mask_idx"))
        .with_columns((pl.col("max_mask_idx") >= 0).alias("detected"))
        .select(["species_id", "filename", "detected"])
    )


def main():
    parser = argparse.ArgumentParser(description="SAM3 plant vs flower prompt agreement analysis")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("~/scratch/plantclef-2026/sam3_outputs").expanduser(),
    )
    parser.add_argument("--top_n", type=int, default=20, help="Number of most-disagreeing species to show")
    args = parser.parse_args()

    data_dir: Path = args.data_dir.expanduser() if not args.data_dir.is_absolute() else args.data_dir

    for prompt in ["plant", "flower"]:
        if not (data_dir / prompt).exists():
            print(f"ERROR: prompt dir not found: {data_dir / prompt}")
            return

    print("Loading 'plant' parquets...")
    plant_df = load_detection_map(data_dir / "plant").rename({"detected": "plant_det"})

    print("Loading 'flower' parquets...")
    flower_df = load_detection_map(data_dir / "flower").rename({"detected": "flower_det"})

    # Full outer join on (species_id, filename)
    joined = plant_df.join(flower_df, on=["species_id", "filename"], how="full", coalesce=True)

    # Fill nulls (images present in only one prompt's output)
    joined = joined.with_columns([
        pl.col("plant_det").fill_null(False),
        pl.col("flower_det").fill_null(False),
    ])

    total = len(joined)

    both = joined.filter(pl.col("plant_det") & pl.col("flower_det")).height
    neither = joined.filter(~pl.col("plant_det") & ~pl.col("flower_det")).height
    plant_only = joined.filter(pl.col("plant_det") & ~pl.col("flower_det")).height
    flower_only = joined.filter(~pl.col("plant_det") & pl.col("flower_det")).height

    print(f"\n{'='*50}")
    print("2x2 Contingency Table (plant vs flower)")
    print(f"{'='*50}")
    print(f"Total images (union): {total:,}\n")
    print(f"  both detected:    {both:>10,}  ({100*both/total:.1f}%)")
    print(f"  neither:          {neither:>10,}  ({100*neither/total:.1f}%)")
    print(f"  plant only:       {plant_only:>10,}  ({100*plant_only/total:.1f}%)")
    print(f"  flower only:      {flower_only:>10,}  ({100*flower_only/total:.1f}%)")
    disagree = plant_only + flower_only
    print(f"\n  disagreement:     {disagree:>10,}  ({100*disagree/total:.1f}%)")

    # Per-species breakdown sorted by disagreement rate
    species_stats = (
        joined.group_by("species_id")
        .agg([
            pl.len().alias("total_images"),
            (pl.col("plant_det") & pl.col("flower_det")).sum().alias("both"),
            (~pl.col("plant_det") & ~pl.col("flower_det")).sum().alias("neither"),
            (pl.col("plant_det") & ~pl.col("flower_det")).sum().alias("plant_only"),
            (~pl.col("plant_det") & pl.col("flower_det")).sum().alias("flower_only"),
        ])
        .with_columns(
            ((pl.col("plant_only") + pl.col("flower_only")) / pl.col("total_images") * 100.0)
            .alias("disagree_pct")
        )
        .sort("disagree_pct", descending=True)
    )

    top_n = min(args.top_n, len(species_stats))
    print(f"\nTop {top_n} most-disagreeing species:")
    header = f"{'species_id':<20} {'images':>8} {'both':>8} {'neither':>8} {'plant_only':>12} {'flower_only':>12} {'disagree%':>11}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for row in species_stats.head(top_n).iter_rows(named=True):
        print(
            f"{str(row['species_id']):<20} {row['total_images']:>8,} {row['both']:>8,} "
            f"{row['neither']:>8,} {row['plant_only']:>12,} {row['flower_only']:>12,} "
            f"{row['disagree_pct']:>10.1f}%"
        )
    print(sep)


if __name__ == "__main__":
    main()
