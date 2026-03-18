"""
Threshold sensitivity analysis for SAM3 inference outputs.

Shows how detection rate and mask count change at different score thresholds
without re-running inference.

Usage:
    UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv uv run --package 001-sam3-pcs \
        python analysis/threshold_sensitivity.py --prompt flower [--data_dir DIR] \
        [--thresholds 0.5 0.6 0.7 0.8 0.9]
"""

import argparse
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(description="SAM3 threshold sensitivity analysis")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("~/scratch/plantclef-2026/sam3_outputs").expanduser(),
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt subdirectory name (e.g. 'flower')")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        metavar="T",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.expanduser() if not args.data_dir.is_absolute() else args.data_dir
    prompt_dir = data_dir / args.prompt

    if not prompt_dir.exists():
        print(f"ERROR: prompt dir not found: {prompt_dir}")
        return

    # Load all parquets
    parquet_files = sorted(prompt_dir.glob("task_*/results.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {prompt_dir}")
        return

    print(f"Loading {len(parquet_files)} parquet files for prompt '{args.prompt}'...")
    df = pl.concat([pl.read_parquet(p) for p in parquet_files])

    total_images = df.select("filename").n_unique()
    print(f"Total unique images: {total_images:,}\n")

    # For each threshold, compute stats
    header = f"{'threshold':>10} {'detected':>12} {'det%':>8} {'total_masks':>14} {'masks/det':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for thresh in sorted(args.thresholds):
        # Images with at least one mask at or above threshold
        detected_images = (
            df.filter(pl.col("mask_idx") >= 0)
            .filter(pl.col("score") >= thresh)
            .select("filename")
            .n_unique()
        )
        det_pct = 100.0 * detected_images / total_images if total_images > 0 else 0.0

        total_masks = df.filter(pl.col("mask_idx") >= 0).filter(pl.col("score") >= thresh).height
        masks_per_det = total_masks / detected_images if detected_images > 0 else 0.0

        print(
            f"{thresh:>10.1f} {detected_images:>12,} {det_pct:>7.1f}% {total_masks:>14,} {masks_per_det:>12.2f}"
        )

    print(sep)
    print("Note: threshold=0.5 is baseline (inference already filtered at this level)")


if __name__ == "__main__":
    main()
