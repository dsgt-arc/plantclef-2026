"""Filter ConvNeXt tile parquets by per-tile vegetation fraction.

Drops rows where veg_frac < threshold. Outputs new tile parquets that
can be passed to ensemble_v2.py via --convnext-scales.

Mechanism in one sentence: drop tiles that are mostly non-vegetation
(substrate, ruler markings, wooden frame) before max_prob aggregation,
so high-confidence non-plant predictions don't dominate the per-class
maximum.

threshold=0.0 reproduces the original tile parquet byte-equivalent
(modulo column order/types; semantically equivalent).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_VEG = (r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub"
               r"\plantclef-2026\user\alper\src\Exp-16_Multi_Species_Discovery"
               r"\outputs\image_audit\per_tile_veg_frac.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--scale", type=int, required=True,
                    help="Tile scale (4, 5, or 6).")
    ap.add_argument("--veg-frac-parquet", default=DEFAULT_VEG)
    ap.add_argument("--veg-threshold", type=float, required=True,
                    help="Drop tiles where veg_frac < threshold.")
    args = ap.parse_args()

    print(f"  reading tile parquet: {args.in_parquet}")
    df = pd.read_parquet(args.in_parquet)
    print(f"    {len(df):,} rows")

    veg = pd.read_parquet(args.veg_frac_parquet)
    veg = veg[veg["scale"] == args.scale][[
        "image_name", "tile_index", "veg_frac"
    ]]
    print(f"  veg-frac for scale {args.scale}: {len(veg):,} rows")

    # tile_index in tile parquet may be offset by scale (production convention
    # adds scale*1000); strip if present.
    if df["tile_index"].max() >= 1000:
        df = df.copy()
        df["tile_index"] = df["tile_index"] - args.scale * 1000

    merged = df.merge(veg, on=["image_name", "tile_index"], how="left")
    n_missing = merged["veg_frac"].isna().sum()
    if n_missing > 0:
        print(f"  WARN: {n_missing} rows had no veg_frac match (kept as if veg_frac=1.0)")
        merged["veg_frac"] = merged["veg_frac"].fillna(1.0)

    keep = merged["veg_frac"] >= args.veg_threshold
    print(f"  threshold {args.veg_threshold}: keeping {keep.sum():,} / {len(keep):,} "
          f"({100*keep.sum()/len(keep):.1f}%)")

    # Per-image fallback: if filtering would leave an image with zero tiles,
    # keep its top-veg tile so the image isn't dropped from downstream.
    merged_keep = merged.copy()
    merged_keep["_keep"] = keep
    img_kept = merged_keep.groupby("image_name")["_keep"].sum()
    zero_imgs = img_kept[img_kept == 0].index.tolist()
    if zero_imgs:
        print(f"  per-image fallback: {len(zero_imgs)} images with 0 kept tiles -> "
              f"restoring top-veg tile each")
        for nm in zero_imgs:
            sub = merged_keep[merged_keep["image_name"] == nm]
            best_idx = sub["veg_frac"].idxmax()
            merged_keep.at[best_idx, "_keep"] = True
    keep = merged_keep["_keep"]
    out = merged[keep].drop(columns=["veg_frac"]).copy()
    print(f"  after fallback: {keep.sum():,} rows")

    Path(os.path.dirname(os.path.abspath(args.out_parquet))).mkdir(
        parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"  wrote {len(out):,} rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
