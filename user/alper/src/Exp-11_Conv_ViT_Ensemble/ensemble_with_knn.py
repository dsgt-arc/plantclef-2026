"""Exp-11: Blend ViT+kNN fused probabilities with ConvNeXt probabilities.

Takes pre-computed ViT+kNN parquet (from Exp-9 pipeline) and ConvNeXt
multi-scale tiles, aggregates ConvNeXt, then blends and generates submissions.

Usage:
  python ensemble_with_knn.py \
    --vit-knn-parquet /path/to/vit_knn_fused.parquet \
    --convnext-scales f1.parquet f2.parquet ... \
    --species-map /path/to/map.json \
    --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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


def load_fused_parquet(path: str) -> Dict[str, np.ndarray]:
    """Load pre-aggregated/fused parquet."""
    df = pq.read_table(path).to_pandas()
    results = {}
    id_col = "image_name" if "image_name" in df.columns else df.columns[0]
    probs_col = "image_probs_f16" if "image_probs_f16" in df.columns else df.columns[1]
    for _, row in df.iterrows():
        probs = np.frombuffer(row[probs_col], dtype=np.float16).copy()
        results[str(row[id_col])] = probs
    print(f"Loaded {len(results)} images from {path}")
    return results


def load_scales(paths):
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
    ap.add_argument("--vit-knn-parquet", required=True,
                    help="Pre-computed ViT+kNN fused parquet from Exp-9")
    ap.add_argument("--convnext-scales", nargs="+", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--cnxt-alpha", type=float, default=0.05,
                    help="Temporal fusion alpha for ConvNeXt")
    ap.add_argument("--weights", default="0.78,0.80,0.82,0.85,0.88,0.90")
    ap.add_argument("--thresholds", default="0.100")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) if args.class_mask_npy else None

    # Load ViT+kNN fused probabilities
    print("=== Loading ViT+kNN fused probabilities ===")
    vit_knn = load_fused_parquet(args.vit_knn_parquet)

    # Aggregate ConvNeXt
    print("\n=== Aggregating ConvNeXt ===")
    cnxt_tiles = load_scales(args.convnext_scales)
    cnxt_probs = aggregate(cnxt_tiles, args.temperature, args.class_mask_npy)
    print(f"  Aggregated {len(cnxt_probs)} images")

    # Temporal fusion for ConvNeXt
    print(f"  Temporal fusion (alpha={args.cnxt_alpha})...")
    cnxt_fused = temporal_fusion(cnxt_probs, args.cnxt_alpha, args.out_dir, "cnxt")
    print(f"  Fused {len(cnxt_fused)} images")

    # Blend and sweep
    common = set(vit_knn.keys()) & set(cnxt_fused.keys())
    print(f"\nViT+kNN: {len(vit_knn)} images")
    print(f"ConvNeXt: {len(cnxt_fused)} images")
    print(f"Common: {len(common)} images")
    print(f"Weights: {weights}")
    print(f"Thresholds: {thresholds}")

    for w in weights:
        ens = {}
        for name in common:
            pv = vit_knn[name].astype(np.float32)
            pc = cnxt_fused[name].astype(np.float32)
            ens[name] = (w * pv + (1 - w) * pc).astype(np.float16)
        tag = f"w{w:.2f}"

        print(f"\n  --- {tag} ---")
        for thr in thresholds:
            csv = os.path.join(args.out_dir, f"ens_knn_{tag}_thr{thr:.3f}_top{args.topk}.csv")
            generate_csv(ens, species_map, class_mask, args.topk, thr, csv)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
