"""Exp-16 Phase 1 — compute per-image per-class vote-share signal.

For each test image, compute `vote_share[c]` = fraction of tiles in which
class c appears in the per-tile top-K (after softmax with the production
T=1.5 temperature). Output is a per-image float16 vector of length 7806,
saved to a parquet file with columns image_name, vote_share_f16.

This is the upstream signal for `apply_vote_rerank.py`, which multiplies
the production ViT prefusion by `(vote_share + epsilon)^beta` to boost
species with broad multi-tile support over single-tile noise spikes.

Mechanism in one sentence: count, per image, how often each species
lands in the per-tile top-K across all tiles in the multi-scale set.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

EXP14 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXP14, "..", "Exp-7_Multiscale_tiling"))
sys.path.insert(0, os.path.join(EXP14, "..", "Exp-8_Quantile_Aggregation_SS"))

from utils.io_bytes import f16_bytes_list_to_f32_matrix  # noqa: E402

NUM_CLASSES = 7806


def softmax_T(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits.astype(np.float32, copy=False) / temperature
    x -= np.max(x, axis=1, keepdims=True)
    ex = np.exp(x, dtype=np.float32)
    return ex / np.sum(ex, axis=1, keepdims=True, dtype=np.float32)


def per_tile_topK_indices(probs: np.ndarray, K: int) -> np.ndarray:
    if K >= probs.shape[1]:
        return np.tile(np.arange(probs.shape[1]), (probs.shape[0], 1))
    idx = np.argpartition(probs, -K, axis=1)[:, -K:]
    return idx


def vote_share_for_image(
    tile_logits: np.ndarray,
    K: int,
    temperature: float,
) -> np.ndarray:
    p = softmax_T(tile_logits, temperature)
    topk = per_tile_topK_indices(p, K)
    counts = np.bincount(topk.ravel(), minlength=NUM_CLASSES)
    return (counts / float(tile_logits.shape[0])).astype(np.float32)


def load_concat_scales(scale_paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in scale_paths:
        t0 = time.time()
        df = pd.read_parquet(p, columns=[
            "image_name", "tile_index", "teacher_logits_f16", "passed_aug_consistency",
        ])
        n_img = df["image_name"].nunique()
        print(f"  loaded {os.path.basename(p)}: "
              f"{len(df):,} tiles, {n_img} images, {time.time()-t0:.1f}s")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    n_img = combined["image_name"].nunique()
    print(f"  combined: {len(combined):,} tiles, {n_img} images")
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tile-dir",
        default=os.path.join(EXP14, "..", "Exp-14_Adaptive_Temporal_Fusion",
                             "pace_downloads", "ViT_classifier"),
    )
    ap.add_argument("--scales", default="4,5,6")
    ap.add_argument("--K", type=int, required=True,
                    help="Per-tile top-K for vote computation (e.g. 5, 10).")
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--aug-filter", action="store_true",
                    help="Restrict to passed_aug_consistency==True (production default).")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    paths = [os.path.join(args.tile_dir, f"scale_{s}x{s}.parquet") for s in scales]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    print("=== Exp-16 Phase 1: compute_vote_share ===")
    print(f"  tile-dir:    {args.tile_dir}")
    print(f"  scales:      {scales}")
    print(f"  K:           {args.K}")
    print(f"  temperature: {args.temperature}")
    print(f"  aug-filter:  {args.aug_filter}")
    print(f"  out:         {args.out_parquet}")

    combined = load_concat_scales(paths)

    if args.aug_filter and "passed_aug_consistency" in combined.columns:
        before = len(combined)
        combined = combined[combined["passed_aug_consistency"] == True]  # noqa: E712
        print(f"  aug-filter: {before:,} -> {len(combined):,} tiles")

    out_rows = []
    t0 = time.time()
    last_print = t0
    grouped = combined.groupby("image_name", sort=True)
    n_groups = grouped.ngroups
    print(f"\n  computing vote_share over {n_groups} images...")
    for i, (name, group) in enumerate(grouped):
        bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(bufs, NUM_CLASSES)
        vs = vote_share_for_image(tile_logits, args.K, args.temperature)
        out_rows.append({
            "image_name": str(name),
            "vote_share_f16": vs.astype(np.float16).tobytes(),
            "n_tiles": int(tile_logits.shape[0]),
        })
        if time.time() - last_print > 5.0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_groups - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{n_groups}  rate={rate:.1f} img/s  eta={eta:.0f}s")
            last_print = time.time()

    out_df = pd.DataFrame(out_rows)
    Path(os.path.dirname(os.path.abspath(args.out_parquet))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"\n  wrote {len(out_df)} rows -> {args.out_parquet}")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
