"""Sweep threshold values on an aggregated/fused parquet and compute diagnostics.

Generates one Kaggle submission CSV per threshold value, then prints a
diagnostics table comparing prediction counts and marginal probabilities
to help prioritise which CSVs to submit (since F1 is Kaggle-only).

CPU-only, runs in seconds.  No re-aggregation needed.
"""
from __future__ import annotations

import argparse
import csv
import os
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

DEFAULT_THRESHOLDS = [
    0.0, 0.005, 0.01, 0.015, 0.02, 0.025,
    0.03,                                     # current baseline
    0.04, 0.05, 0.07, 0.10, 0.15,
]


def _count_predictions_per_image(csv_path: str) -> List[int]:
    """Read a Kaggle CSV and return the number of species per row."""
    counts: List[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            species_str = row[1].strip()
            if species_str in ("[]", ""):
                counts.append(0)
            else:
                # "[1234, 5678, ...]" -> count commas + 1
                counts.append(species_str.count(",") + 1)
    return counts


def _compute_marginal_probs(
    parquet_path: str,
    class_mask: Optional[np.ndarray],
    topk: int,
    threshold: float,
    id_col: str,
    probs_col: str,
) -> List[float]:
    """Return the probability of the lowest-confidence kept prediction per image."""
    t = pq.read_table(parquet_path, columns=[id_col, probs_col]).to_pandas()
    marginals: List[float] = []
    for _, r in t.iterrows():
        p = f16_bytes_to_f32_vector(r[probs_col], NUM_CLASSES)
        if class_mask is not None:
            p = p.copy()
            p[~class_mask] = 0.0
        k = min(topk, p.shape[0])
        idx = np.argpartition(p, -k)[-k:]
        idx = idx[np.argsort(-p[idx])]
        kept_probs = [float(p[i]) for i in idx if p[i] >= threshold]
        if kept_probs:
            marginals.append(kept_probs[-1])  # lowest among kept
        else:
            marginals.append(0.0)
    return marginals


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep threshold values and compute diagnostics."
    )
    ap.add_argument("--in-parquet", required=True,
                    help="Aggregated or fused parquet (image_name + image_probs_f16)")
    ap.add_argument("--out-dir", required=True,
                    help="Directory for output CSVs and diagnostics")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--id-col", default="image_name")
    ap.add_argument("--probs-col", default="image_probs_f16")
    ap.add_argument("--thresholds", default=None,
                    help="Comma-separated threshold values (default: built-in 12-point grid)")

    args = ap.parse_args()

    thresholds = (
        [float(t.strip()) for t in args.thresholds.split(",")]
        if args.thresholds
        else DEFAULT_THRESHOLDS
    )

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = (
        load_class_mask_npy(args.class_mask_npy, NUM_CLASSES)
        if args.class_mask_npy
        else None
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Generate CSVs and collect diagnostics ──
    baseline_counts: Optional[List[int]] = None
    rows = []

    print(f"=== Threshold sweep ({len(thresholds)} values) ===")
    print(f"Input:  {args.in_parquet}")
    print(f"Output: {args.out_dir}")
    print(f"TopK:   {args.topk}  (fixed)")
    print()

    for thr in sorted(thresholds):
        csv_name = f"thr_{thr:.3f}_top{args.topk}.csv"
        csv_path = os.path.join(args.out_dir, csv_name)

        n_rows = generate_submission_csv(
            in_parquet=args.in_parquet,
            out_csv=csv_path,
            species_map=species_map,
            num_classes=NUM_CLASSES,
            class_mask=class_mask,
            topk=args.topk,
            threshold=thr,
            min_k=1,
            id_col=args.id_col,
            probs_col=args.probs_col,
        )

        counts = _count_predictions_per_image(csv_path)
        arr = np.array(counts, dtype=np.float64)

        marginals = _compute_marginal_probs(
            args.in_parquet, class_mask, args.topk, thr,
            args.id_col, args.probs_col,
        )
        avg_marginal = float(np.mean(marginals)) if marginals else 0.0

        if baseline_counts is None and abs(thr - 0.03) < 1e-9:
            baseline_counts = counts

        delta_str = "—"
        if baseline_counts is not None:
            delta = float(np.mean(arr)) - float(np.mean(baseline_counts))
            delta_str = f"{delta:+.2f}"

        rows.append({
            "thr": thr,
            "n_rows": n_rows,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "delta": delta_str,
            "avg_marginal": avg_marginal,
            "csv": csv_name,
        })

    # ── Print summary table ──
    header = (
        f"{'Threshold':>10} | {'Avg preds':>9} | {'Median':>6} | "
        f"{'Min':>3} | {'Max':>3} | {'Delta vs 0.03':>13} | "
        f"{'Avg marginal':>12} | CSV"
    )
    sep = "-" * len(header)

    lines = [
        "",
        "=== Threshold Sweep Diagnostics ===",
        f"Parquet: {args.in_parquet}",
        f"TopK:   {args.topk}",
        "",
        header,
        sep,
    ]

    for r in rows:
        line = (
            f"{r['thr']:>10.3f} | {r['mean']:>9.2f} | {r['median']:>6.1f} | "
            f"{r['min']:>3d} | {r['max']:>3d} | {r['delta']:>13s} | "
            f"{r['avg_marginal']:>12.5f} | {r['csv']}"
        )
        lines.append(line)

    lines.append(sep)
    lines.append("")

    report = "\n".join(lines)
    print(report)

    diag_path = os.path.join(args.out_dir, "diagnostics.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Diagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
