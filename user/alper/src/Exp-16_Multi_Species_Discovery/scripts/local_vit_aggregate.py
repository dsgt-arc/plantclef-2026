"""Locally aggregate ViT classifier tile parquets at beta=1.0 (no kNN).

For each test image:
  1. Concat tile rows across scale_{4,5,6} parquets (optionally filtered).
  2. Apply aug_consistency filter (production default).
  3. Per-tile softmax of teacher_logits at T=1.5.
  4. Optional veg_frac filter: drop tiles with veg_frac < threshold.
  5. Optional image_topk: keep only top-K classes per image post-aggregation.
  6. Apply geo mask.
  7. Max across tiles -> per-image probability vector.

Output: parquet with image_name, image_probs_f16, num_nonzero — same
schema as the production knn_bM0.85_bS1.00_T1.50_prefusion.parquet so it
plugs directly into ensemble_v2 via --vit-prefusion-parquet.

beta=1.0 means no kNN blending (production uses 0.85 multi-visit / 1.00
single). The kNN delta in production is ~+0.005 F1, but it's orthogonal
to whether the veg_frac filter helps. The veg_frac filter test is a
delta between two beta=1.0 runs (filtered vs unfiltered), so the
absence of kNN affects both equally.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EXP14 = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-14_Adaptive_Temporal_Fusion"))
EXP8 = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-8_Quantile_Aggregation_SS"))
sys.path.insert(0, EXP14)
sys.path.insert(0, EXP8)
from utils.io_bytes import f16_bytes_list_to_f32_matrix  # noqa: E402

NUM_CLASSES = 7806


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale4-parquet", required=True)
    ap.add_argument("--scale5-parquet", required=True)
    ap.add_argument("--scale6-parquet", required=True)
    ap.add_argument("--veg-frac-parquet", default=None,
                    help="If given, drop tiles below veg-threshold per scale.")
    ap.add_argument("--veg-threshold", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--geo-mask", default=None)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--aug-filter", action="store_true",
                    help="Filter tiles to passed_aug_consistency==True (production default).")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    print("=" * 72)
    print("Local ViT aggregator (no-kNN, beta=1.0)")
    print("=" * 72)
    print(f"  veg_threshold:   {args.veg_threshold}")
    print(f"  temperature:     {args.temperature}")
    print(f"  image_topk:      {args.image_topk}")
    print(f"  aug_filter:      {args.aug_filter}")

    geo_mask = None
    if args.geo_mask:
        geo_mask = np.load(args.geo_mask).astype(np.bool_)
        print(f"  geo mask passes: {geo_mask.sum()} / {NUM_CLASSES}")

    veg_df: Optional[pd.DataFrame] = None
    if args.veg_frac_parquet and args.veg_threshold > 0:
        veg_df = pd.read_parquet(args.veg_frac_parquet)
        print(f"  loaded veg lookup: {len(veg_df):,} entries")

    frames = []
    for scale, path in [(4, args.scale4_parquet),
                        (5, args.scale5_parquet),
                        (6, args.scale6_parquet)]:
        t0 = time.time()
        df = pd.read_parquet(path, columns=[
            "image_name", "tile_index", "teacher_logits_f16",
            "passed_aug_consistency"
        ])
        df["scale"] = scale
        if df["tile_index"].max() >= 1000:
            df = df.copy()
            df["tile_index"] = df["tile_index"] - scale * 1000
        if args.aug_filter:
            before = len(df)
            df = df[df["passed_aug_consistency"] == True]  # noqa: E712
            print(f"    scale {scale}: aug-filter {before:,} -> {len(df):,}")
        if veg_df is not None:
            sub_veg = veg_df[veg_df["scale"] == scale][[
                "image_name", "tile_index", "veg_frac"
            ]]
            df = df.merge(sub_veg, on=["image_name", "tile_index"], how="left")
            df["veg_frac"] = df["veg_frac"].fillna(1.0)
            before = len(df)
            df = df[df["veg_frac"] >= args.veg_threshold]
            print(f"    scale {scale}: veg-filter {before:,} -> {len(df):,}")
        else:
            print(f"    scale {scale}: {len(df):,} tiles ({time.time()-t0:.1f}s)")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  combined: {len(combined):,} tiles, "
          f"{combined['image_name'].nunique()} images")

    # Per-image fallback: if all tiles were filtered out across all scales,
    # restore that image's top-veg tile (any scale) so the image isn't dropped.
    if veg_df is not None:
        all_imgs = set()
        for path in [args.scale4_parquet, args.scale5_parquet, args.scale6_parquet]:
            tmp = pd.read_parquet(path, columns=["image_name"])
            all_imgs.update(tmp["image_name"].unique().tolist())
        kept_imgs = set(combined["image_name"].unique().tolist())
        missing = all_imgs - kept_imgs
        if missing:
            print(f"\n  per-image fallback: {len(missing)} images with 0 tiles -> "
                  f"restoring top-veg tile each")
            restored_rows = []
            for nm in missing:
                # Find best veg tile across all scales for this image
                best = veg_df[veg_df["image_name"] == nm].nlargest(1, "veg_frac")
                if best.empty:
                    continue
                row = best.iloc[0]
                scale = int(row["scale"])
                tidx = int(row["tile_index"])
                # Look up the corresponding teacher_logits
                src_path = {4: args.scale4_parquet, 5: args.scale5_parquet,
                            6: args.scale6_parquet}[scale]
                tile_df = pd.read_parquet(src_path, columns=[
                    "image_name", "tile_index", "teacher_logits_f16",
                    "passed_aug_consistency"
                ])
                if tile_df["tile_index"].max() >= 1000:
                    tile_df = tile_df.copy()
                    tile_df["tile_index"] = tile_df["tile_index"] - scale * 1000
                tile_match = tile_df[(tile_df["image_name"] == nm) &
                                      (tile_df["tile_index"] == tidx)]
                if not tile_match.empty:
                    rec = tile_match.iloc[0].to_dict()
                    rec["scale"] = scale
                    rec["veg_frac"] = float(row["veg_frac"])
                    restored_rows.append(rec)
            if restored_rows:
                combined = pd.concat(
                    [combined, pd.DataFrame(restored_rows)],
                    ignore_index=True,
                )
            print(f"  after fallback: {len(combined):,} tiles, "
                  f"{combined['image_name'].nunique()} images")

    out_rows = []
    t0 = time.time()
    last_print = t0
    grouped = combined.groupby("image_name", sort=True)
    n_groups = grouped.ngroups
    print(f"\n  aggregating {n_groups} images...")
    for i, (name, group) in enumerate(grouped):
        bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(bufs, NUM_CLASSES)
        x = tile_logits.astype(np.float32, copy=False) / args.temperature
        x -= np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        p_softmax = ex / np.sum(ex, axis=1, keepdims=True, dtype=np.float32)
        p_image = np.max(p_softmax, axis=0).astype(np.float32)
        np.clip(p_image, 0.0, 1.0, out=p_image)
        if args.image_topk and args.image_topk > 0 and args.image_topk < NUM_CLASSES:
            k = args.image_topk
            idx = np.argpartition(p_image, -k)[-k:]
            mask = np.zeros(len(p_image), dtype=np.bool_)
            mask[idx] = True
            p_image[~mask] = 0.0
        if geo_mask is not None:
            p_image[~geo_mask] = 0.0
        nz = int(np.count_nonzero(p_image))
        out_rows.append({
            "image_name": str(name),
            "image_probs_f16": p_image.astype(np.float16).tobytes(),
            "num_nonzero": nz,
        })
        if time.time() - last_print > 5.0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    {i+1}/{n_groups}  rate={rate:.1f} img/s")
            last_print = time.time()

    out_df = pd.DataFrame(out_rows)
    Path(os.path.dirname(os.path.abspath(args.out_parquet))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"\n  wrote {len(out_df)} rows -> {args.out_parquet}")
    print(f"  total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
