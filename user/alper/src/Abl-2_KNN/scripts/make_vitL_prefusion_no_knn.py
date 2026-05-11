"""No-kNN fork of Exp-14C/make_vitL_prefusion.py.

Skips the FAISS retriever entirely. Per-image pipeline becomes:
  classifier softmax (T=1.5) -> max_prob across tiles -> image_topk=64
  -> geo mask -> save as pre-fusion parquet.

The original script has a `if b < 1.0: blend else: skip` branch at
beta_multi=1.0, so beta_multi=1.0 is mathematically equivalent to this
script. The difference: this fork skips FAISS loading + per-tile kNN
queries entirely (saves ~1.5–2 GPU/CPU-h per run and removes the FAISS
index file as a dependency).

Output schema is identical to make_vitL_prefusion.py — consumable by
ensemble_v2.py (Exp-14) and ensemble_vit_only.py (Abl-1) via
--vit-prefusion-parquet.

Usage:
  python make_vitL_prefusion_no_knn.py \
    --vit-scales scale_4x4.parquet scale_5x5.parquet scale_6x6.parquet scale_7x7.parquet \
    --class-mask-npy .../sw_europe_mask_bool.npy \
    --out-dir .../abl2_prefusion \
    --temperature 1.5 \
    --image-topk 64 \
    --tag vitL_noknn
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABL2 = os.path.dirname(SCRIPT_DIR)
SRC = os.path.dirname(ABL2)
EXP8_DIR = os.path.join(SRC, "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, EXP8_DIR)

from aggregate_tiles import save_aggregated  # noqa: E402
from utils.io_bytes import f16_bytes_list_to_f32_matrix  # noqa: E402

NUM_CLASSES = 7806


def load_vit_scales(paths):
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        print(f"  {os.path.basename(p)}: {len(df)} tiles, "
              f"{df['image_name'].nunique()} images")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} tiles, "
          f"{combined['image_name'].nunique()} images")
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-scales", nargs="+", required=True,
                    help="Per-tile ViT-L parquets (one per scale).")
    ap.add_argument("--class-mask-npy", default=None,
                    help="Geo mask .npy (sw_europe_mask_bool.npy).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--tag", default="vitL_noknn")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    geo_mask = (np.load(args.class_mask_npy).astype(np.bool_)
                if args.class_mask_npy else None)
    if geo_mask is not None:
        print(f"[init] geo mask: {geo_mask.sum()}/{len(geo_mask)} classes pass")

    print(f"\n=== Loading ViT-L tiles ===")
    tile_data = load_vit_scales(args.vit_scales)

    print(f"\n=== Per-image: classifier softmax -> max_prob -> top{args.image_topk} ===")
    t0 = time.time()
    results: Dict[str, np.ndarray] = {}
    n_images = 0
    image_names = tile_data["image_name"].unique()
    for name, group in tile_data.groupby("image_name"):
        logits_bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, NUM_CLASSES)
        x = tile_logits.astype(np.float32) / args.temperature
        x -= np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        p_softmax = ex / np.sum(ex, axis=1, keepdims=True)

        # No kNN — pure classifier
        p_image = np.max(p_softmax, axis=0).astype(np.float32)
        np.clip(p_image, 0.0, 1.0, out=p_image)
        p_image[p_image < 1e-6] = 0.0

        tk = min(args.image_topk, len(p_image))
        idx = np.argpartition(p_image, -tk)[-tk:]
        mask_arr = np.zeros(len(p_image), dtype=np.bool_)
        mask_arr[idx] = True
        p_image[~mask_arr] = 0.0

        if geo_mask is not None:
            p_image[~geo_mask] = 0.0

        results[str(name)] = p_image.astype(np.float16)
        n_images += 1
        if n_images % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_images / elapsed
            eta = (len(image_names) - n_images) / rate
            print(f"  {n_images}/{len(image_names)}  {rate:.1f} img/s  "
                  f"ETA {eta:.0f}s")

    out_path = os.path.join(args.out_dir, f"{args.tag}_prefusion.parquet")
    save_aggregated(results, out_path)
    print(f"\n  wrote {out_path}  ({len(results)} images, {time.time() - t0:.1f}s)")
    print(f"\n=== Done. No-kNN pre-fusion parquet ready for downstream. ===")


if __name__ == "__main__":
    main()
