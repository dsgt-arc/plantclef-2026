"""
Summary statistics for SAM3 inference outputs.

Usage:
    UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv uv run --package 001-sam3-pcs \
        python analysis/summary_stats.py [--data_dir DIR] [--prompt PROMPT]
"""

import argparse
import os
from pathlib import Path

import polars as pl


def load_task_parquets(prompt_dir: Path) -> list[tuple[str, pl.DataFrame]]:
    tasks = sorted(prompt_dir.glob("task_*/results.parquet"))
    results = []
    for p in tasks:
        task_name = p.parent.name
        try:
            df = pl.read_parquet(p)
            results.append((task_name, df))
        except Exception as e:
            print(f"  WARNING: could not read {p}: {e}")
    return results


def task_stats(task_name: str, df: pl.DataFrame) -> dict:
    total_images = df.select("filename").n_unique()
    no_det = df.filter(pl.col("mask_idx") == -1).select("filename").n_unique()
    with_det = total_images - no_det
    det_pct = 100.0 * with_det / total_images if total_images > 0 else 0.0
    mask_rows = df.filter(pl.col("mask_idx") >= 0)
    total_masks = len(mask_rows)
    masks_per_det = total_masks / with_det if with_det > 0 else 0.0
    return {
        "task": task_name,
        "images": total_images,
        "with_det": with_det,
        "no_det": no_det,
        "det_pct": det_pct,
        "masks_per_det": masks_per_det,
    }


def print_table(rows: list[dict], size_mb: dict[str, float]) -> None:
    header = f"{'task':<14} {'images':>10} {'w/det':>10} {'no-det':>10} {'det%':>8} {'masks/det':>10} {'size MB':>10}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        mb = size_mb.get(r["task"], 0.0)
        print(
            f"{r['task']:<14} {r['images']:>10,} {r['with_det']:>10,} {r['no_det']:>10,} "
            f"{r['det_pct']:>7.1f}% {r['masks_per_det']:>10.2f} {mb:>10.1f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="SAM3 output summary statistics")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("~/scratch/plantclef-2026/sam3_outputs").expanduser(),
    )
    parser.add_argument("--prompt", type=str, default=None, help="Filter to a specific prompt subdirectory")
    args = parser.parse_args()

    data_dir: Path = args.data_dir.expanduser() if not args.data_dir.is_absolute() else args.data_dir

    if not data_dir.exists():
        print(f"ERROR: data_dir does not exist: {data_dir}")
        return

    # Discover prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    if not prompts:
        print(f"No prompt subdirectories found in {data_dir}")
        return

    for prompt in prompts:
        prompt_dir = data_dir / prompt
        if not prompt_dir.exists():
            print(f"WARNING: prompt dir not found: {prompt_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        task_list = load_task_parquets(prompt_dir)
        if not task_list:
            print("  No task parquets found.")
            continue

        rows = []
        size_mb_map = {}
        totals = {"images": 0, "with_det": 0, "no_det": 0}

        for task_name, df in task_list:
            r = task_stats(task_name, df)
            rows.append(r)
            # File size
            p = prompt_dir / task_name / "results.parquet"
            size_mb_map[task_name] = p.stat().st_size / 1e6

            totals["images"] += r["images"]
            totals["with_det"] += r["with_det"]
            totals["no_det"] += r["no_det"]

        print_table(rows, size_mb_map)

        total_masks_per_det = (
            totals["with_det"] and
            sum(r["masks_per_det"] * r["with_det"] for r in rows) / totals["with_det"]
        )
        total_det_pct = 100.0 * totals["with_det"] / totals["images"] if totals["images"] > 0 else 0.0
        total_mb = sum(size_mb_map.values())

        print(
            f"{'TOTAL':<14} {totals['images']:>10,} {totals['with_det']:>10,} {totals['no_det']:>10,} "
            f"{total_det_pct:>7.1f}% {total_masks_per_det:>10.2f} {total_mb:>10.1f}"
        )


if __name__ == "__main__":
    main()
