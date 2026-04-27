"""Phase-2 post-extraction sanity check.

For each scale, verify that the photometric extraction parquet:
  - has the same row count as the baseline
  - covers the same (image_name, tile_index) pairs
  - has max_prob median within ±0.10 of baseline (sign of distribution collapse otherwise)

Usage:
    python diagnostics/postextract_sanity.py \
        --baseline-dir /local/path/exp9b_baseline_test/ \
        --photo-dir   /local/path/exp15_photo_unsharp_test/ \
        --scales 4,5,6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def check_scale(baseline_path: str, photo_path: str, scale: int) -> bool:
    if not os.path.exists(baseline_path):
        print(f"  ! baseline missing: {baseline_path}")
        return False
    if not os.path.exists(photo_path):
        print(f"  ! photo missing: {photo_path}")
        return False
    base = pd.read_parquet(baseline_path, columns=["image_name", "tile_index", "max_prob"])
    photo = pd.read_parquet(photo_path, columns=["image_name", "tile_index", "max_prob"])
    print(f"  scale {scale}x{scale}:")
    print(f"    baseline rows={len(base)}  photo rows={len(photo)}")
    if len(base) != len(photo):
        print(f"    ROW COUNT MISMATCH (FAIL)")
        return False
    base_keys = set(zip(base["image_name"], base["tile_index"]))
    photo_keys = set(zip(photo["image_name"], photo["tile_index"]))
    if base_keys != photo_keys:
        diff = (base_keys - photo_keys) | (photo_keys - base_keys)
        print(f"    KEY MISMATCH: {len(diff)} differing tile keys (FAIL)")
        return False
    median_shift = float(photo["max_prob"].median() - base["max_prob"].median())
    print(f"    max_prob median: base={base['max_prob'].median():.4f} "
          f"photo={photo['max_prob'].median():.4f}  shift={median_shift:+.4f}")
    if abs(median_shift) > 0.10:
        print(f"    MEDIAN SHIFT > 0.10 (FAIL)")
        return False
    print(f"    PASS")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--photo-dir", required=True)
    ap.add_argument("--scales", default="4,5,6")
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    results = []
    for scale in scales:
        bp = os.path.join(args.baseline_dir, f"scale_{scale}x{scale}.parquet")
        pp = os.path.join(args.photo_dir, f"scale_{scale}x{scale}.parquet")
        results.append(check_scale(bp, pp, scale))

    if all(results):
        print("\n  All scales PASS sanity check.")
        sys.exit(0)
    else:
        print("\n  Sanity check FAILED — do not proceed to Phase 3.")
        sys.exit(2)


if __name__ == "__main__":
    main()
