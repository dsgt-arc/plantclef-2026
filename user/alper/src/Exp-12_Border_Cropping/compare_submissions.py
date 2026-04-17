"""Compare submission CSVs across border crop levels.

Since F1 requires Kaggle submission, this script analyzes how predictions
change between crop levels to give early signal before submitting.

Metrics computed:
  - Mean/median/max predictions per image
  - Species overlap (Jaccard) between baseline and cropped
  - Number of images with changed predictions
  - Top species added/removed by cropping

Usage:
  python compare_submissions.py \
    --baseline-csv /path/to/baseline/scheme_A_*.csv \
    --crop-csvs /path/to/crop_3pct/scheme_A_*.csv \
                /path/to/crop_5pct/scheme_A_*.csv \
                /path/to/crop_7pct/scheme_A_*.csv

  Or point to parquet directories for richer analysis:
  python compare_submissions.py \
    --baseline-parquet /path/to/baseline/scheme_A_*_prefusion.parquet \
    --crop-parquets /path/to/crop_3pct/*.parquet \
                    /path/to/crop_5pct/*.parquet \
                    /path/to/crop_7pct/*.parquet
"""
from __future__ import annotations

import argparse
import ast
import os
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def load_submission_csv(csv_path: str) -> Dict[str, Set[int]]:
    """Load a Kaggle submission CSV into {quadrat_id: set(species_ids)}."""
    df = pd.read_csv(csv_path)
    results = {}
    for _, row in df.iterrows():
        qid = str(row["quadrat_id"])
        species_str = str(row["species_ids"])
        try:
            species = set(ast.literal_eval(species_str))
        except (ValueError, SyntaxError):
            species = set()
        results[qid] = species
    return results


def compare_predictions(
    baseline: Dict[str, Set[int]],
    cropped: Dict[str, Set[int]],
    label: str,
) -> Dict[str, float]:
    """Compare two sets of predictions and return summary metrics."""
    common_qids = set(baseline.keys()) & set(cropped.keys())
    if not common_qids:
        print(f"  WARNING: No common quadrat IDs between baseline and {label}")
        return {}

    jaccards = []
    n_changed = 0
    n_added_total = 0
    n_removed_total = 0
    baseline_counts = []
    cropped_counts = []
    added_species: Counter = Counter()
    removed_species: Counter = Counter()

    for qid in sorted(common_qids):
        b_set = baseline[qid]
        c_set = cropped[qid]
        baseline_counts.append(len(b_set))
        cropped_counts.append(len(c_set))

        if b_set == c_set:
            jaccards.append(1.0)
            continue

        n_changed += 1
        union = b_set | c_set
        inter = b_set & c_set
        jaccard = len(inter) / len(union) if union else 1.0
        jaccards.append(jaccard)

        added = c_set - b_set
        removed = b_set - c_set
        n_added_total += len(added)
        n_removed_total += len(removed)
        for s in added:
            added_species[s] += 1
        for s in removed:
            removed_species[s] += 1

    n_total = len(common_qids)
    pct_changed = n_changed / n_total * 100

    print(f"\n  --- {label} ---")
    print(f"  Images compared:     {n_total}")
    print(f"  Images changed:      {n_changed} ({pct_changed:.1f}%)")
    print(f"  Mean Jaccard:        {np.mean(jaccards):.4f}")
    print(f"  Median Jaccard:      {np.median(jaccards):.4f}")
    print(f"  Baseline avg preds:  {np.mean(baseline_counts):.2f}")
    print(f"  Cropped avg preds:   {np.mean(cropped_counts):.2f}")
    print(f"  Species added:       {n_added_total} total across {len(added_species)} unique species")
    print(f"  Species removed:     {n_removed_total} total across {len(removed_species)} unique species")

    if added_species:
        top_added = added_species.most_common(5)
        print(f"  Top added species:   {top_added}")
    if removed_species:
        top_removed = removed_species.most_common(5)
        print(f"  Top removed species: {top_removed}")

    return {
        "label": label,
        "n_changed": n_changed,
        "pct_changed": pct_changed,
        "mean_jaccard": np.mean(jaccards),
        "median_jaccard": np.median(jaccards),
        "n_added": n_added_total,
        "n_removed": n_removed_total,
        "baseline_avg_preds": np.mean(baseline_counts),
        "cropped_avg_preds": np.mean(cropped_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare submission CSVs across border crop levels"
    )
    ap.add_argument("--baseline-csv", required=True,
                    help="Baseline submission CSV (no crop)")
    ap.add_argument("--crop-csvs", nargs="+", required=True,
                    help="Cropped submission CSVs (one per crop level)")
    ap.add_argument("--crop-labels", nargs="+", default=None,
                    help="Labels for each crop CSV (default: filenames)")
    args = ap.parse_args()

    print("=== Exp-12: Border Crop Submission Comparison ===")
    print(f"Baseline: {args.baseline_csv}")

    baseline = load_submission_csv(args.baseline_csv)
    print(f"  Loaded {len(baseline)} quadrats")

    labels = args.crop_labels or [
        os.path.basename(os.path.dirname(c)) or os.path.basename(c)
        for c in args.crop_csvs
    ]

    all_results = []
    for csv_path, label in zip(args.crop_csvs, labels):
        print(f"\nLoading {csv_path}...")
        cropped = load_submission_csv(csv_path)
        print(f"  Loaded {len(cropped)} quadrats")
        result = compare_predictions(baseline, cropped, label)
        if result:
            all_results.append(result)

    if all_results:
        print("\n\n=== SUMMARY TABLE ===")
        print(f"{'Crop Level':<20} {'Changed':>8} {'% Changed':>10} "
              f"{'Jaccard':>8} {'Added':>7} {'Removed':>8}")
        print("-" * 70)
        for r in all_results:
            print(f"{r['label']:<20} {r['n_changed']:>8} "
                  f"{r['pct_changed']:>9.1f}% "
                  f"{r['mean_jaccard']:>8.4f} "
                  f"{r['n_added']:>7} {r['n_removed']:>8}")

    print("\n\nInterpretation:")
    print("  High Jaccard (>0.95) = cropping barely changes predictions -> likely no F1 impact")
    print("  Low Jaccard (<0.80)  = significant prediction changes -> worth submitting to Kaggle")
    print("  More removed than added = cropping is removing false positives (good)")
    print("  More added than removed = cropping is introducing new predictions (check quality)")


if __name__ == "__main__":
    main()
