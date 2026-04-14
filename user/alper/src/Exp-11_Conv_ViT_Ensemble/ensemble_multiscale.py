"""Exp-11: Full-pipeline ensemble — multi-scale + temporal fusion + blend.

Aggregates ConvNeXt and ViT tiles independently through the full pipeline
(multi-scale max_prob + temporal fusion), then blends at the post-processed
probability level.

Usage:
  python ensemble_multiscale.py \
    --convnext-4x4 /path/to/cnxt/scale_4x4.parquet \
    --convnext-5x5 /path/to/cnxt/scale_5x5.parquet \
    --convnext-6x6 /path/to/cnxt/scale_6x6.parquet \
    --vit-4x4 /path/to/vit/scale_4x4.parquet \
    --vit-5x5 /path/to/vit/scale_5x5.parquet \
    --vit-6x6 /path/to/vit/scale_6x6.parquet \
    --species-map /path/to/class_to_species_id.json \
    --class-mask-npy /path/to/sw_europe_mask_bool.npy \
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
import pyarrow.parquet as pq

# Add Exp-8 to path for reuse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP8_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, EXP8_DIR)

from utils.aggregator import MaxProbabilityAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes
from aggregate_tiles import save_aggregated
from temporal_fusion import (
    load_quadrat_probs,
    group_by_location,
    build_location_profiles,
    fuse_all,
)
from generate_submission import load_species_map, load_class_mask_npy

NUM_CLASSES = 7806


def load_and_combine_scales(scale_paths: Dict[int, str]) -> pd.DataFrame:
    """Load parquet files for each scale and combine into one DataFrame."""
    frames = []
    for scale, path in sorted(scale_paths.items()):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scale {scale}x{scale} not found: {path}")
        df = pd.read_parquet(path)
        df["scale"] = scale
        n_images = df["image_name"].nunique()
        print(f"  Scale {scale}x{scale}: {len(df)} tiles, {n_images} images")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} tiles, {combined['image_name'].nunique()} images")
    return combined


def aggregate_multiscale_max_prob(
    tile_data: pd.DataFrame,
    temperature: float = 1.5,
    geo_mask_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate multi-scale tiles per image using max_prob."""
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

    return results


def apply_temporal_fusion(
    image_probs: Dict[str, np.ndarray],
    alpha: float,
    out_dir: str,
    tag: str,
) -> Dict[str, np.ndarray]:
    """Apply prior_blend temporal fusion."""
    # Save to temp parquet for temporal fusion functions
    temp_path = os.path.join(out_dir, f"_temp_{tag}_prefusion.parquet")
    save_aggregated(image_probs, temp_path)

    quadrat_probs = load_quadrat_probs(temp_path)
    location_groups = group_by_location(list(quadrat_probs.keys()))
    location_profiles = build_location_profiles(
        quadrat_probs, location_groups, NUM_CLASSES
    )
    fused_probs = fuse_all(
        quadrat_probs,
        location_groups,
        location_profiles,
        strategy="prior_blend",
        params={"alpha": alpha},
    )

    # Clean up temp file
    os.remove(temp_path)

    # Convert to f16
    return {qid: p.astype(np.float16) for qid, p in fused_probs.items()}


def generate_submission(
    image_probs: Dict[str, np.ndarray],
    species_map: list,
    class_mask: Optional[np.ndarray],
    topk: int,
    threshold: float,
    out_csv: str,
    min_k: int = 1,
) -> None:
    """Generate Kaggle submission CSV."""
    rows = []
    for image_name, probs in sorted(image_probs.items()):
        p = probs.astype(np.float32)
        if class_mask is not None:
            p[~class_mask] = 0.0
        top_indices = np.argsort(p)[::-1][:topk]
        selected = [idx for idx in top_indices if p[idx] >= threshold]
        if len(selected) < min_k:
            selected = list(top_indices[:min_k])
        species_ids = [species_map[idx] for idx in selected]
        quadrat_id = os.path.splitext(image_name)[0]
        rows.append({"quadrat_id": quadrat_id, "species_ids": str(species_ids)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"    Wrote {len(df)} rows to {os.path.basename(out_csv)}")


def blend_probabilities(
    probs_a: Dict[str, np.ndarray],
    probs_b: Dict[str, np.ndarray],
    weight_a: float,
) -> Dict[str, np.ndarray]:
    """P = weight_a * A + (1-weight_a) * B."""
    common = set(probs_a.keys()) & set(probs_b.keys())
    results = {}
    for name in common:
        p_a = probs_a[name].astype(np.float32)
        p_b = probs_b[name].astype(np.float32)
        results[name] = (weight_a * p_a + (1 - weight_a) * p_b).astype(np.float16)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Exp-11: Full-pipeline multi-scale ensemble"
    )

    # ConvNeXt scale parquets
    ap.add_argument("--convnext-4x4", required=True)
    ap.add_argument("--convnext-5x5", required=True)
    ap.add_argument("--convnext-6x6", required=True)

    # ViT scale parquets
    ap.add_argument("--vit-4x4", required=True)
    ap.add_argument("--vit-5x5", required=True)
    ap.add_argument("--vit-6x6", required=True)

    # Mapping and masking
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)

    # Output
    ap.add_argument("--out-dir", required=True)

    # Params
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Temporal fusion alpha (Exp-8 best: 0.05)")
    ap.add_argument("--weights", default="0.0,0.4,0.5,0.6,0.7,0.8,1.0",
                    help="ViT weights to sweep")
    ap.add_argument("--thresholds", default="0.04,0.075,0.100",
                    help="Thresholds to sweep")
    ap.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = (
        load_class_mask_npy(args.class_mask_npy, NUM_CLASSES)
        if args.class_mask_npy else None
    )

    # ---- Aggregate ConvNeXt ----
    print("\n" + "=" * 60)
    print("AGGREGATING ConvNeXt (4x4 + 5x5 + 6x6)")
    print("=" * 60)
    cnxt_scales = {
        4: args.convnext_4x4,
        5: args.convnext_5x5,
        6: args.convnext_6x6,
    }
    cnxt_tiles = load_and_combine_scales(cnxt_scales)
    cnxt_probs = aggregate_multiscale_max_prob(
        cnxt_tiles, args.temperature, args.class_mask_npy
    )
    print(f"  Aggregated {len(cnxt_probs)} images")

    # Temporal fusion
    print(f"  Applying temporal fusion (alpha={args.alpha})...")
    cnxt_fused = apply_temporal_fusion(
        cnxt_probs, args.alpha, args.out_dir, "convnext"
    )
    save_aggregated(cnxt_fused, os.path.join(args.out_dir, "convnext_multiscale_fused.parquet"))
    print(f"  ConvNeXt fused: {len(cnxt_fused)} images")

    # ---- Aggregate ViT ----
    print("\n" + "=" * 60)
    print("AGGREGATING ViT (4x4 + 5x5 + 6x6)")
    print("=" * 60)
    vit_scales = {
        4: args.vit_4x4,
        5: args.vit_5x5,
        6: args.vit_6x6,
    }
    vit_tiles = load_and_combine_scales(vit_scales)
    vit_probs = aggregate_multiscale_max_prob(
        vit_tiles, args.temperature, args.class_mask_npy
    )
    print(f"  Aggregated {len(vit_probs)} images")

    # Temporal fusion
    print(f"  Applying temporal fusion (alpha={args.alpha})...")
    vit_fused = apply_temporal_fusion(
        vit_probs, args.alpha, args.out_dir, "vit"
    )
    save_aggregated(vit_fused, os.path.join(args.out_dir, "vit_multiscale_fused.parquet"))
    print(f"  ViT fused: {len(vit_fused)} images")

    # ---- Ensemble sweep ----
    print("\n" + "=" * 60)
    print("ENSEMBLE SWEEP")
    print(f"Weights (ViT): {weights}")
    print(f"Thresholds: {thresholds}")
    print("=" * 60)

    common = set(vit_fused.keys()) & set(cnxt_fused.keys())
    print(f"Common images: {len(common)}")

    for w in weights:
        if w == 0.0:
            ensemble = cnxt_fused
            tag = "convnext_only"
        elif w == 1.0:
            ensemble = vit_fused
            tag = "vit_only"
        else:
            ensemble = blend_probabilities(vit_fused, cnxt_fused, weight_a=w)
            tag = f"w{w:.1f}"

        print(f"\n  --- {tag} ---")
        for thr in thresholds:
            csv_path = os.path.join(
                args.out_dir,
                f"ms_submission_{tag}_a{args.alpha}_thr{thr:.3f}_top{args.topk}.csv"
            )
            generate_submission(
                ensemble, species_map, class_mask,
                topk=args.topk, threshold=thr, out_csv=csv_path,
            )

    print(f"\n=== Done. Results in {args.out_dir} ===")


if __name__ == "__main__":
    main()
