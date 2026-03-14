"""Adaptive per-image thresholding for PlantCLEF species prediction.

Instead of a fixed global threshold + topk, each strategy examines the
shape of the per-image probability curve to decide how many species to
predict.  Generates one Kaggle CSV per (strategy, parameter) combination
and prints a diagnostics table for submission prioritisation.

CPU-only, runs in under a minute for all 34 configurations.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from generate_submission import (
    f16_bytes_to_f32_vector,
    format_species_list,
    load_class_mask_npy,
    load_species_map,
)

NUM_CLASSES = 7806


# ────────────────────────────────────────────────────────────────────
# Strategy functions
#
# Each takes a *descending-sorted* probability array and returns the
# number of species to keep (raw K, before min/max clamping).
# ────────────────────────────────────────────────────────────────────

def strategy_max_gap(probs_sorted: np.ndarray, max_candidates: int = 50) -> int:
    """Cut at the largest probability gap within the top-N candidates."""
    n = min(max_candidates, len(probs_sorted))
    if n <= 1:
        return 1
    segment = probs_sorted[:n]
    gaps = segment[:-1] - segment[1:]
    gap_idx = int(np.argmax(gaps))
    return gap_idx + 1  # keep everything before the gap


def strategy_relative_ratio(probs_sorted: np.ndarray, ratio: float = 0.1) -> int:
    """Keep species whose probability is >= ratio * max probability."""
    if len(probs_sorted) == 0 or probs_sorted[0] <= 0:
        return 1
    cutoff = ratio * probs_sorted[0]
    count = int(np.searchsorted(-probs_sorted, -cutoff))  # first index below cutoff
    return max(count, 1)


def strategy_elbow(
    probs_sorted: np.ndarray,
    smooth_window: int = 3,
    max_candidates: int = 30,
) -> int:
    """Find the elbow (maximum curvature) in the sorted probability curve.

    Uses a Kneedle-like approach: normalise x,y to [0,1], find the point
    of maximum perpendicular distance from the line connecting endpoints.
    """
    n = min(max_candidates, len(probs_sorted))
    if n <= 2:
        return 1

    y = probs_sorted[:n].astype(np.float64)

    # Optional smoothing
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        y = np.convolve(y, kernel, mode="same")

    x = np.arange(n, dtype=np.float64)

    # Normalise to [0, 1]
    x_range = x[-1] - x[0]
    y_range = y[0] - y[-1]
    if x_range == 0 or y_range <= 0:
        return 1
    x_norm = (x - x[0]) / x_range
    y_norm = (y - y[-1]) / y_range

    # Perpendicular distance from line (0,1) -> (1,0)
    # Line: x + y = 1, distance = |x + y - 1| / sqrt(2)
    distances = np.abs(x_norm + y_norm - 1.0)
    elbow_idx = int(np.argmax(distances))

    return max(elbow_idx + 1, 1)


def strategy_fixed_threshold(probs_sorted: np.ndarray, threshold: float = 0.03) -> int:
    """Baseline: count species above a fixed probability threshold."""
    count = int(np.searchsorted(-probs_sorted, -threshold))
    return max(count, 1)


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_image_probs(
    parquet_path: str,
    id_col: str,
    probs_col: str,
    num_classes: int,
    class_mask: Optional[np.ndarray],
) -> List[Tuple[str, np.ndarray]]:
    """Load parquet and return list of (quadrat_id, masked_probs)."""
    t = pq.read_table(parquet_path, columns=[id_col, probs_col]).to_pandas()
    images: List[Tuple[str, np.ndarray]] = []
    for _, r in t.iterrows():
        qid = os.path.splitext(str(r[id_col]))[0]
        p = f16_bytes_to_f32_vector(r[probs_col], num_classes)
        if class_mask is not None:
            p = p.copy()
            p[~class_mask] = 0.0
        images.append((qid, p))
    return images


# ────────────────────────────────────────────────────────────────────
# CSV writer (bypasses generate_submission_csv to avoid its own topk)
# ────────────────────────────────────────────────────────────────────

def write_adaptive_csv(
    predictions: List[Tuple[str, List[int]]],
    out_path: str,
) -> int:
    """Write Kaggle submission CSV from adaptive predictions."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, species_ids in predictions:
            w.writerow([qid, format_species_list(species_ids)])
    return len(predictions)


# ────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    strategy: str
    params_str: str
    csv_path: str
    avg_k: float = 0.0
    median_k: float = 0.0
    min_k: int = 0
    max_k: int = 0
    std_k: float = 0.0
    avg_marginal: float = 0.0
    avg_gap_at_cut: float = 0.0


# ────────────────────────────────────────────────────────────────────
# Sweep runner
# ────────────────────────────────────────────────────────────────────

def _build_sweep_configs(strategies: str) -> List[Tuple[str, str, dict]]:
    """Build list of (strategy_name, param_label, param_dict)."""
    want = set(s.strip() for s in strategies.split(",")) if strategies != "all" else None
    configs: List[Tuple[str, str, dict]] = []

    if want is None or "max_gap" in want:
        for n in [10, 15, 20, 25, 30, 40, 50]:
            configs.append(("max_gap", f"N={n}", {"max_candidates": n}))

    if want is None or "relative_ratio" in want:
        for r in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]:
            configs.append(("relative_ratio", f"ratio={r:.2f}", {"ratio": r}))

    if want is None or "elbow" in want:
        for sw in [1, 3, 5]:
            for nc in [20, 30, 50]:
                configs.append(("elbow", f"smooth={sw}_N={nc}",
                                {"smooth_window": sw, "max_candidates": nc}))

    if want is None or "fixed_threshold" in want:
        for t in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]:
            configs.append(("fixed_threshold", f"thr={t:.3f}", {"threshold": t}))

    return configs


STRATEGY_FN = {
    "max_gap": strategy_max_gap,
    "relative_ratio": strategy_relative_ratio,
    "elbow": strategy_elbow,
    "fixed_threshold": strategy_fixed_threshold,
}


def run_sweep(
    images: List[Tuple[str, np.ndarray]],
    species_map: List[int],
    out_dir: str,
    min_k: int,
    max_k: int,
    strategies: str,
) -> List[SweepResult]:
    """Run all strategy x param combinations and return results."""
    configs = _build_sweep_configs(strategies)
    results: List[SweepResult] = []

    print(f"Running {len(configs)} configurations on {len(images)} images...")
    print()

    for strategy_name, params_str, params in configs:
        fn = STRATEGY_FN[strategy_name]
        predictions: List[Tuple[str, List[int]]] = []
        ks: List[int] = []
        marginals: List[float] = []
        gaps: List[float] = []

        for qid, probs in images:
            # Sort descending, track original indices
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]

            # Strategy decides raw K
            raw_k = fn(sorted_probs, **params)
            k = max(min_k, min(raw_k, max_k))

            # Collect top-K species
            top_indices = sorted_idx[:k]
            species_ids = [species_map[int(i)] for i in top_indices]
            predictions.append((qid, species_ids))

            # Diagnostics
            ks.append(k)
            marginals.append(float(sorted_probs[k - 1]))  # lowest kept
            if k < len(sorted_probs):
                gaps.append(float(sorted_probs[k - 1] - sorted_probs[k]))
            else:
                gaps.append(0.0)

        # Write CSV
        csv_name = f"{strategy_name}_{params_str.replace('=', '').replace(' ', '_')}.csv"
        csv_path = os.path.join(out_dir, csv_name)
        write_adaptive_csv(predictions, csv_path)

        k_arr = np.array(ks, dtype=np.float64)
        result = SweepResult(
            strategy=strategy_name,
            params_str=params_str,
            csv_path=csv_name,
            avg_k=float(np.mean(k_arr)),
            median_k=float(np.median(k_arr)),
            min_k=int(np.min(k_arr)),
            max_k=int(np.max(k_arr)),
            std_k=float(np.std(k_arr)),
            avg_marginal=float(np.mean(marginals)),
            avg_gap_at_cut=float(np.mean(gaps)),
        )
        results.append(result)
        print(f"  {strategy_name:20s} {params_str:20s} -> avg_k={result.avg_k:.1f}  "
              f"med={result.median_k:.0f}  std={result.std_k:.1f}  "
              f"marginal={result.avg_marginal:.4f}  gap={result.avg_gap_at_cut:.4f}  "
              f"-> {csv_name}")

    return results


# ────────────────────────────────────────────────────────────────────
# Diagnostics output
# ────────────────────────────────────────────────────────────────────

def format_diagnostics(results: List[SweepResult], parquet_path: str) -> str:
    """Format results into a readable diagnostics table."""
    header = (
        f"{'Strategy':20s} | {'Params':20s} | {'Avg K':>6s} | {'Med K':>5s} | "
        f"{'Min':>3s} | {'Max':>3s} | {'Std':>5s} | "
        f"{'Avg Marginal':>12s} | {'Avg Gap@Cut':>11s} | CSV"
    )
    sep = "-" * len(header)

    lines = [
        "",
        "=== Exp-6: Adaptive Per-Image Thresholding Diagnostics ===",
        f"Parquet: {parquet_path}",
        "",
        header,
        sep,
    ]

    for r in results:
        line = (
            f"{r.strategy:20s} | {r.params_str:20s} | {r.avg_k:>6.1f} | "
            f"{r.median_k:>5.0f} | {r.min_k:>3d} | {r.max_k:>3d} | "
            f"{r.std_k:>5.1f} | {r.avg_marginal:>12.5f} | "
            f"{r.avg_gap_at_cut:>11.5f} | {r.csv_path}"
        )
        lines.append(line)

    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def write_diagnostics_csv(results: List[SweepResult], out_path: str) -> None:
    """Write diagnostics as machine-readable CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "params", "avg_k", "median_k", "min_k", "max_k",
            "std_k", "avg_marginal", "avg_gap_at_cut", "csv",
        ])
        for r in results:
            w.writerow([
                r.strategy, r.params_str, f"{r.avg_k:.2f}",
                f"{r.median_k:.0f}", r.min_k, r.max_k,
                f"{r.std_k:.2f}", f"{r.avg_marginal:.6f}",
                f"{r.avg_gap_at_cut:.6f}", r.csv_path,
            ])


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Adaptive per-image thresholding sweep for PlantCLEF."
    )
    ap.add_argument("--in-parquet", required=True,
                    help="Aggregated/fused parquet (image_name + image_probs_f16)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for CSVs + diagnostics")
    ap.add_argument("--species-map", required=True,
                    help="Path to class_to_species_id.json")
    ap.add_argument("--class-mask-npy", default=None,
                    help="Geo-species boolean mask (.npy)")
    ap.add_argument("--id-col", default="image_name")
    ap.add_argument("--probs-col", default="image_probs_f16")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--min-k", type=int, default=1,
                    help="Minimum predictions per image (default: 1)")
    ap.add_argument("--max-k", type=int, default=25,
                    help="Maximum predictions per image (default: 25)")
    ap.add_argument("--strategies", default="all",
                    help="Comma-separated: max_gap,relative_ratio,elbow,fixed_threshold "
                         "or 'all' (default: all)")

    args = ap.parse_args()

    # Load data once
    print(f"=== Exp-6: Adaptive Per-Image Thresholding ===")
    print(f"Input:      {args.in_parquet}")
    print(f"Output:     {args.out_dir}")
    print(f"Strategies: {args.strategies}")
    print(f"K range:    [{args.min_k}, {args.max_k}]")
    print()

    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = (
        load_class_mask_npy(args.class_mask_npy, args.num_classes)
        if args.class_mask_npy
        else None
    )

    t0 = time.time()
    print("Loading image probabilities...")
    images = load_image_probs(
        args.in_parquet, args.id_col, args.probs_col,
        args.num_classes, class_mask,
    )
    print(f"Loaded {len(images)} images in {time.time() - t0:.1f}s")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run sweep
    results = run_sweep(
        images, species_map, args.out_dir,
        args.min_k, args.max_k, args.strategies,
    )

    # Write diagnostics
    report = format_diagnostics(results, args.in_parquet)
    print(report)

    diag_txt = os.path.join(args.out_dir, "diagnostics.txt")
    with open(diag_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Diagnostics (text): {diag_txt}")

    diag_csv = os.path.join(args.out_dir, "diagnostics.csv")
    write_diagnostics_csv(results, diag_csv)
    print(f"Diagnostics (csv):  {diag_csv}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({len(results)} configurations)")


if __name__ == "__main__":
    main()
