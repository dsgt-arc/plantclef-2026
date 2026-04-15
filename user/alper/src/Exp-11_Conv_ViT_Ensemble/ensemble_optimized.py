"""Exp-11: Optimized ensemble — each model uses its best scale scheme.

ConvNeXt: 3+4+5+6+7, ViT: 4+5+6
Both aggregated with max_prob, T=1.5, temporal fusion (alpha=0.05)
Then blended at probability level and swept over weights + thresholds.

Usage:
  python ensemble_optimized.py \
    --convnext-scales f1.parquet f2.parquet ... \
    --vit-scales f1.parquet f2.parquet ... \
    --species-map /path/to/map.json \
    --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP8_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, EXP8_DIR)

from utils.aggregator import MaxProbabilityAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix
from aggregate_tiles import save_aggregated
from temporal_fusion import (
    load_quadrat_probs, group_by_location,
    build_location_profiles, fuse_all,
)
from generate_submission import load_species_map, load_class_mask_npy

NUM_CLASSES = 7806


def load_scales(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        print(f"  {os.path.basename(p)}: {len(df)} tiles, {df['image_name'].nunique()} images")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} tiles, {combined['image_name'].nunique()} images")
    return combined


def aggregate(tile_data, temperature, geo_mask_path):
    aggregator = MaxProbabilityAggregator(
        num_classes=NUM_CLASSES, temperature=temperature, image_topk=64)
    geo_mask = np.load(geo_mask_path).astype(np.bool_) if geo_mask_path else None

    results = {}
    for name, group in tile_data.groupby("image_name"):
        logits = f16_bytes_list_to_f32_matrix(
            group["teacher_logits_f16"].tolist(), num_classes=NUM_CLASSES)
        p = aggregator.aggregate(logits)
        if geo_mask is not None:
            p32 = p.astype(np.float32)
            p32[~geo_mask] = 0.0
            p = p32.astype(np.float16)
        results[str(name)] = p
    return results


def temporal_fusion(probs, alpha, out_dir, tag):
    tmp = os.path.join(out_dir, f"_tmp_{tag}.parquet")
    save_aggregated(probs, tmp)
    qp = load_quadrat_probs(tmp)
    lg = group_by_location(list(qp.keys()))
    lp = build_location_profiles(qp, lg, NUM_CLASSES)
    fused = fuse_all(qp, lg, lp, strategy="prior_blend", params={"alpha": alpha})
    os.remove(tmp)
    return {k: v.astype(np.float16) for k, v in fused.items()}


def generate_csv(probs, species_map, class_mask, topk, threshold, out_csv):
    rows = []
    for name, p in sorted(probs.items()):
        p = p.astype(np.float32)
        if class_mask is not None:
            p[~class_mask] = 0.0
        top_idx = np.argsort(p)[::-1][:topk]
        selected = [i for i in top_idx if p[i] >= threshold]
        if not selected:
            selected = [top_idx[0]]
        rows.append({
            "quadrat_id": os.path.splitext(name)[0],
            "species_ids": str([species_map[i] for i in selected])
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    {os.path.basename(out_csv)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convnext-scales", nargs="+", required=True)
    ap.add_argument("--vit-scales", nargs="+", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--weights", default="0.75,0.78,0.80,0.82,0.85")
    ap.add_argument("--thresholds", default="0.090,0.100,0.110,0.120")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--prefix", default="ens",
                    help="Prefix for output CSV filenames")
    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) if args.class_mask_npy else None

    # Aggregate ConvNeXt
    print("\n" + "=" * 60)
    print("CONVNEXT")
    print("=" * 60)
    cnxt_tiles = load_scales(args.convnext_scales)
    cnxt_probs = aggregate(cnxt_tiles, args.temperature, args.class_mask_npy)
    print(f"  Aggregated {len(cnxt_probs)} images")
    cnxt_fused = temporal_fusion(cnxt_probs, args.alpha, args.out_dir, "cnxt")
    print(f"  Fused {len(cnxt_fused)} images")

    # Aggregate ViT
    print("\n" + "=" * 60)
    print("VIT")
    print("=" * 60)
    vit_tiles = load_scales(args.vit_scales)
    vit_probs = aggregate(vit_tiles, args.temperature, args.class_mask_npy)
    print(f"  Aggregated {len(vit_probs)} images")
    vit_fused = temporal_fusion(vit_probs, args.alpha, args.out_dir, "vit")
    print(f"  Fused {len(vit_fused)} images")

    # Sweep
    common = set(vit_fused.keys()) & set(cnxt_fused.keys())
    print(f"\nCommon images: {len(common)}")
    print(f"Weights: {weights}")
    print(f"Thresholds: {thresholds}")

    for w in weights:
        if w == 0.0:
            ens = cnxt_fused; tag = "convnext_only"
        elif w == 1.0:
            ens = vit_fused; tag = "vit_only"
        else:
            ens = {}
            for name in common:
                pv = vit_fused[name].astype(np.float32)
                pc = cnxt_fused[name].astype(np.float32)
                ens[name] = (w * pv + (1 - w) * pc).astype(np.float16)
            tag = f"w{w:.2f}"

        print(f"\n  --- {tag} ---")
        for thr in thresholds:
            csv = os.path.join(args.out_dir, f"{args.prefix}_{tag}_thr{thr:.3f}_top{args.topk}.csv")
            generate_csv(ens, species_map, class_mask, args.topk, thr, csv)

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
