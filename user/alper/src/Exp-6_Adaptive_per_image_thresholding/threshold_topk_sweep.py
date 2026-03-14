"""Joint threshold x topk sweep in the region where both parameters interact.

At high threshold (>=0.04), topk rarely binds.
At low threshold + low topk, topk becomes the constraint.
This sweep finds the optimal combination.

CPU-only, seconds per config.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq

from generate_submission import (
    f16_bytes_to_f32_vector,
    generate_submission_csv,
    load_class_mask_npy,
    load_species_map,
)

NUM_CLASSES = 7806

DEFAULT_TOPKS = [3, 4, 5, 6, 7]
DEFAULT_THRESHOLDS = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]


def _count_predictions(csv_path: str) -> List[int]:
    """Count species predictions per row in a Kaggle CSV."""
    counts: List[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            s = row[1].strip()
            if s in ("[]", ""):
                counts.append(0)
            else:
                counts.append(s.count(",") + 1)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint threshold x topk sweep.")
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--id-col", default="image_name")
    ap.add_argument("--probs-col", default="image_probs_f16")
    ap.add_argument("--topks", default=None,
                    help="Comma-separated topk values (default: 3,4,5,6,7)")
    ap.add_argument("--thresholds", default=None,
                    help="Comma-separated threshold values (default: 0.02-0.05)")
    args = ap.parse_args()

    topks = (
        [int(k.strip()) for k in args.topks.split(",")]
        if args.topks else DEFAULT_TOPKS
    )
    thresholds = (
        [float(t.strip()) for t in args.thresholds.split(",")]
        if args.thresholds else DEFAULT_THRESHOLDS
    )

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = (
        load_class_mask_npy(args.class_mask_npy, NUM_CLASSES)
        if args.class_mask_npy else None
    )

    os.makedirs(args.out_dir, exist_ok=True)

    n_combos = len(topks) * len(thresholds)
    print(f"=== Threshold x TopK Sweep ===")
    print(f"Input:      {args.in_parquet}")
    print(f"Output:     {args.out_dir}")
    print(f"TopK:       {topks}")
    print(f"Thresholds: {[f'{t:.3f}' for t in thresholds]}")
    print(f"Total:      {n_combos} combinations")
    print()

    t0 = time.time()
    results = []

    for topk in sorted(topks):
        for thr in sorted(thresholds):
            csv_name = f"thr{thr:.3f}_top{topk}.csv"
            csv_path = os.path.join(args.out_dir, csv_name)

            n_rows = generate_submission_csv(
                in_parquet=args.in_parquet,
                out_csv=csv_path,
                species_map=species_map,
                num_classes=NUM_CLASSES,
                class_mask=class_mask,
                topk=topk,
                threshold=thr,
                min_k=1,
                id_col=args.id_col,
                probs_col=args.probs_col,
            )

            counts = _count_predictions(csv_path)
            arr = np.array(counts, dtype=np.float64)

            results.append({
                "topk": topk,
                "thr": thr,
                "avg_k": float(np.mean(arr)),
                "median_k": float(np.median(arr)),
                "min_k": int(np.min(arr)),
                "max_k": int(np.max(arr)),
                "csv": csv_name,
            })

    # Print results as a grid
    header = (
        f"{'TopK':>4} | {'Threshold':>9} | {'Avg K':>6} | {'Med K':>5} | "
        f"{'Min':>3} | {'Max':>3} | CSV"
    )
    sep = "-" * len(header)

    lines = [
        "",
        "=== Threshold x TopK Sweep Results ===",
        f"Parquet: {args.in_parquet}",
        "",
        header,
        sep,
    ]

    for r in results:
        line = (
            f"{r['topk']:>4} | {r['thr']:>9.3f} | {r['avg_k']:>6.1f} | "
            f"{r['median_k']:>5.0f} | {r['min_k']:>3d} | {r['max_k']:>3d} | "
            f"{r['csv']}"
        )
        lines.append(line)

    lines.append(sep)

    # Print a compact grid view: rows=threshold, cols=topk, cells=avg_k
    lines.append("")
    lines.append("=== Avg K Grid (rows=threshold, cols=topk) ===")
    col_header = "         " + "".join(f"  top{k:>2}" for k in sorted(topks))
    lines.append(col_header)
    for thr in sorted(thresholds):
        row_vals = []
        for topk in sorted(topks):
            match = [r for r in results if r["topk"] == topk and abs(r["thr"] - thr) < 1e-9]
            row_vals.append(f"{match[0]['avg_k']:>6.1f}" if match else "   N/A")
        lines.append(f"thr={thr:.3f}" + "".join(row_vals))

    lines.append("")

    report = "\n".join(lines)
    print(report)

    diag_path = os.path.join(args.out_dir, "diagnostics.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Diagnostics: {diag_path}")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s ({n_combos} combinations)")


if __name__ == "__main__":
    main()
