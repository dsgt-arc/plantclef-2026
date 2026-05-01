"""Merge ViT + ConvNeXt tiles: blend logits, keep ViT embeddings, save parquet."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-scales", nargs="+", required=True)
    ap.add_argument("--cnxt-scales", nargs="+", required=True)
    ap.add_argument("--blend-weight", type=float, required=True,
                    help="ViT weight (e.g., 0.78)")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    # Load ViT tiles
    print("Loading ViT tiles...")
    vit_frames = []
    for p in args.vit_scales:
        df = pd.read_parquet(p)
        print(f"  {os.path.basename(p)}: {len(df)} tiles")
        vit_frames.append(df)
    vit_df = pd.concat(vit_frames, ignore_index=True)

    # Load ConvNeXt tiles
    print("Loading ConvNeXt tiles...")
    cnxt_frames = []
    for p in args.cnxt_scales:
        df = pd.read_parquet(p)
        print(f"  {os.path.basename(p)}: {len(df)} tiles")
        cnxt_frames.append(df)
    cnxt_df = pd.concat(cnxt_frames, ignore_index=True)

    print(f"ViT: {len(vit_df)} tiles, {vit_df['image_name'].nunique()} images")
    print(f"CNxt: {len(cnxt_df)} tiles, {cnxt_df['image_name'].nunique()} images")

    # Merge on (image_name, tile_index)
    print("Merging...")
    merged = vit_df.merge(
        cnxt_df[["image_name", "tile_index", "teacher_logits_f16"]],
        on=["image_name", "tile_index"],
        suffixes=("_vit", "_cnxt"),
        how="inner",
    )
    print(f"Merged: {len(merged)} tiles, {merged['image_name'].nunique()} images")

    # Blend logits (vectorized)
    w = args.blend_weight
    print(f"Blending logits (w_vit={w:.2f})...")
    t0 = time.time()

    vit_logits = np.stack([
        np.frombuffer(b, dtype=np.float16).astype(np.float32)
        for b in merged["teacher_logits_f16_vit"].values
    ])
    cnxt_logits = np.stack([
        np.frombuffer(b, dtype=np.float16).astype(np.float32)
        for b in merged["teacher_logits_f16_cnxt"].values
    ])
    blended = (w * vit_logits + (1 - w) * cnxt_logits).astype(np.float16)
    blended_bytes = [blended[i].tobytes() for i in range(len(blended))]
    del vit_logits, cnxt_logits, blended

    print(f"  Blending done in {time.time() - t0:.1f}s")

    # Build result
    result = pd.DataFrame({
        "image_name": merged["image_name"].values,
        "tile_index": merged["tile_index"].values,
        "embedding_orig_f16": merged["embedding_orig_f16"].values,
        "teacher_logits_f16": blended_bytes,
        "max_prob": merged["max_prob"].values,
        "passed_aug_consistency": [True] * len(merged),
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    result.to_parquet(args.out_parquet, compression="snappy", index=False)
    print(f"Saved {len(result)} tiles to {args.out_parquet}")


if __name__ == "__main__":
    main()
