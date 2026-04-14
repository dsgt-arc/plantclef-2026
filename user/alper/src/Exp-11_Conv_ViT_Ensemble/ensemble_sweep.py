"""Exp-11: Ensemble ConvNeXt + ViT at the probability level.

Workflow:
  1. Aggregate ConvNeXt 4x4 tiles → image-level probabilities
  2. Aggregate ViT 4x4 tiles → image-level probabilities (or load pre-aggregated)
  3. Blend: P = w * P_vit + (1-w) * P_convnext
  4. Apply temporal fusion, geo_mask, threshold → submission CSV

Usage:
  python ensemble_sweep.py \
    --convnext-tiles /path/to/convnext/scale_4x4 \
    --vit-tiles /path/to/vit/scale_4x4 \
    --species-map /path/to/class_to_species_id.json \
    --class-mask-npy /path/to/sw_europe_mask_bool.npy \
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
import pyarrow as pa
import pyarrow.parquet as pq

# Add Exp-8 to path for reuse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP8_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, EXP8_DIR)

from aggregate_tiles import (
    AggregationConfig,
    load_per_tile_parquet,
    run_aggregation,
    save_aggregated,
)
from utils.aggregator import MaxProbabilityAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes

NUM_CLASSES = 7806


def aggregate_tiles_max_prob(
    tile_path: str,
    temperature: float = 1.5,
    geo_mask_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load tiles and aggregate using max_prob at given temperature.

    tile_path can be a directory or a single parquet file.
    """
    print(f"Loading tiles from {tile_path}")
    if os.path.isfile(tile_path):
        tile_data = pd.read_parquet(tile_path)
    else:
        tile_data = load_per_tile_parquet(tile_path)
    print(f"  Loaded {len(tile_data)} tiles for {tile_data['image_name'].nunique()} images")

    config = AggregationConfig(
        temperature=temperature,
        aug_filter=False,
        tile_filter_k=0,
        geo_mask_path=geo_mask_path,
        image_topk=64,
        num_classes=NUM_CLASSES,
    )

    # Override the default aggregator to use MaxProbability
    aggregator = MaxProbabilityAggregator(
        num_classes=NUM_CLASSES,
        temperature=temperature,
        image_topk=64,
    )

    geo_mask = None
    if geo_mask_path:
        geo_mask = np.load(geo_mask_path).astype(np.bool_)

    results = {}
    for name, group in tile_data.groupby("image_name"):
        logits_bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, num_classes=NUM_CLASSES)
        p_image = aggregator.aggregate(tile_logits)

        if geo_mask is not None:
            p_f32 = p_image.astype(np.float32)
            p_f32[~geo_mask] = 0.0
            p_image = p_f32.astype(np.float16)

        results[str(name)] = p_image

    print(f"  Aggregated {len(results)} images")
    return results


def load_aggregated_parquet(path: str) -> Dict[str, np.ndarray]:
    """Load pre-aggregated parquet into {image_name: probs} dict."""
    df = pq.read_table(path).to_pandas()
    results = {}
    for _, row in df.iterrows():
        probs = np.frombuffer(row["image_probs_f16"], dtype=np.float16).copy()
        results[str(row["image_name"])] = probs
    print(f"Loaded {len(results)} images from {path}")
    return results


def blend_probabilities(
    probs_a: Dict[str, np.ndarray],
    probs_b: Dict[str, np.ndarray],
    weight_a: float,
) -> Dict[str, np.ndarray]:
    """Blend: P = weight_a * A + (1 - weight_a) * B for common images."""
    common = set(probs_a.keys()) & set(probs_b.keys())
    if len(common) < len(probs_a):
        print(f"  Warning: {len(probs_a) - len(common)} images only in A")
    if len(common) < len(probs_b):
        print(f"  Warning: {len(probs_b) - len(common)} images only in B")

    results = {}
    for name in common:
        p_a = probs_a[name].astype(np.float32)
        p_b = probs_b[name].astype(np.float32)
        blended = weight_a * p_a + (1 - weight_a) * p_b
        results[name] = blended.astype(np.float16)

    return results


def generate_submission(
    image_probs: Dict[str, np.ndarray],
    species_map: list,
    class_mask: Optional[np.ndarray],
    topk: int,
    threshold: float,
    out_csv: str,
    min_k: int = 1,
) -> None:
    """Generate Kaggle submission CSV from image probabilities."""
    rows = []
    for image_name, probs in sorted(image_probs.items()):
        p = probs.astype(np.float32)

        if class_mask is not None:
            p[~class_mask] = 0.0

        # Get top-k indices above threshold
        top_indices = np.argsort(p)[::-1][:topk]
        selected = [idx for idx in top_indices if p[idx] >= threshold]

        if len(selected) < min_k:
            selected = list(top_indices[:min_k])

        species_ids = [species_map[idx] for idx in selected]

        quadrat_id = os.path.splitext(image_name)[0]
        rows.append({
            "quadrat_id": quadrat_id,
            "species_ids": str(species_ids),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {len(df)} rows to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Exp-11: ConvNeXt + ViT Ensemble Sweep")

    # Input sources (tiles or pre-aggregated)
    ap.add_argument("--convnext-tiles", help="ConvNeXt per-tile parquet directory")
    ap.add_argument("--convnext-agg", help="Pre-aggregated ConvNeXt parquet (alternative)")
    ap.add_argument("--vit-tiles", help="ViT per-tile parquet directory")
    ap.add_argument("--vit-agg", help="Pre-aggregated ViT parquet (alternative)")

    # Mapping and masking
    ap.add_argument("--species-map", required=True, help="class_to_species_id.json")
    ap.add_argument("--class-mask-npy", help="sw_europe_mask_bool.npy")

    # Output
    ap.add_argument("--out-dir", required=True, help="Output directory")

    # Aggregation params
    ap.add_argument("--temperature", type=float, default=1.5)

    # Sweep params
    ap.add_argument("--weights", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="Comma-separated ViT weights to sweep")
    ap.add_argument("--thresholds", default="0.05,0.075,0.100,0.125,0.150",
                    help="Comma-separated thresholds to sweep")
    ap.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # Load species map
    with open(args.species_map, "r") as f:
        species_map = json.load(f)
    print(f"Loaded species map: {len(species_map)} classes")

    # Load class mask
    class_mask = None
    if args.class_mask_npy:
        class_mask = np.load(args.class_mask_npy).astype(np.bool_)
        print(f"Loaded class mask: {class_mask.sum()}/{len(class_mask)} allowed")

    # Aggregate or load ConvNeXt probabilities
    if args.convnext_agg:
        cnxt_probs = load_aggregated_parquet(args.convnext_agg)
    elif args.convnext_tiles:
        cnxt_probs = aggregate_tiles_max_prob(args.convnext_tiles, args.temperature)
        save_aggregated(cnxt_probs, os.path.join(args.out_dir, "convnext_agg.parquet"))
    else:
        raise ValueError("Provide --convnext-tiles or --convnext-agg")

    # Aggregate or load ViT probabilities
    if args.vit_agg:
        vit_probs = load_aggregated_parquet(args.vit_agg)
    elif args.vit_tiles:
        vit_probs = aggregate_tiles_max_prob(args.vit_tiles, args.temperature)
        save_aggregated(vit_probs, os.path.join(args.out_dir, "vit_agg.parquet"))
    else:
        raise ValueError("Provide --vit-tiles or --vit-agg")

    print(f"\nConvNeXt: {len(cnxt_probs)} images")
    print(f"ViT: {len(vit_probs)} images")
    print(f"Common: {len(set(cnxt_probs.keys()) & set(vit_probs.keys()))} images")

    # Sweep
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n=== Ensemble Sweep ===")
    print(f"Weights (ViT): {weights}")
    print(f"Thresholds: {thresholds}")

    for w in weights:
        if w == 0.0:
            # ConvNeXt only
            ensemble = cnxt_probs
            tag = "convnext_only"
        elif w == 1.0:
            # ViT only
            ensemble = vit_probs
            tag = "vit_only"
        else:
            ensemble = blend_probabilities(vit_probs, cnxt_probs, weight_a=w)
            tag = f"w{w:.1f}"

        # Save aggregated ensemble parquet
        agg_path = os.path.join(args.out_dir, f"ensemble_{tag}.parquet")
        save_aggregated(ensemble, agg_path)

        for thr in thresholds:
            csv_name = f"submission_{tag}_thr{thr:.3f}_top{args.topk}.csv"
            csv_path = os.path.join(args.out_dir, csv_name)
            print(f"\n  w={w:.1f}, thr={thr:.3f}:")
            generate_submission(
                ensemble, species_map, class_mask,
                topk=args.topk, threshold=thr, out_csv=csv_path,
            )

    print(f"\n=== Done. Results in {args.out_dir} ===")


if __name__ == "__main__":
    main()
