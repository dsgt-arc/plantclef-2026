"""Exp-8: Quantile aggregation sweep on multi-scale tiles.

Extends Exp-7 by sweeping aggregation methods (max_prob, quantile at various q
values) on the best multi-scale scheme (E: 4x4+5x5+6x6).

Quantile aggregation replaces max(probs_across_tiles) with
np.quantile(probs_across_tiles, q). This keeps most of max_prob's advantage
while being more robust to single spurious high-confidence false positive tiles.

Pipeline:
  1. Load + combine tiles from relevant scale directories
  2. Aggregate with configurable method (max_prob or quantile @ various q)
  3. Temporal fusion (prior_blend, alpha=0.6)
  4. Generate CSVs at multiple thresholds

Usage:
  python multiscale_aggregate.py \
    --scale4-dir /path/to/phase1_tiles_test_jpeg85 \
    --scale5-dir /path/to/scale_5x5 \
    --scale6-dir /path/to/scale_6x6 \
    --model-path /path/to/vit_base_patch14_reg4_dinov2.pth \
    --species-map /path/to/species_map.json \
    --class-mask-npy /path/to/sw_europe_mask_bool.npy \
    --out-dir /path/to/output \
    --schemes E \
    --agg-methods max_prob,quantile \
    --agg-quantiles 0.75,0.80,0.85,0.90,0.95
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aggregate_tiles import (
    AggregationConfig,
    load_per_tile_parquet,
    save_aggregated,
)
from generate_submission import (
    generate_submission_csv,
    load_class_mask_npy,
    load_species_map,
)
from temporal_fusion import (
    build_location_profiles,
    fuse_all,
    group_by_location,
    load_quadrat_probs,
)
from utils.aggregator import create_aggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes

NUM_CLASSES = 7806

# --------------------------------------------------------------------------
# Scheme definitions
# --------------------------------------------------------------------------

SCHEMES = {
    "A": {
        "label": "4x4 (baseline)",
        "scales": [4],
        "tiles": 16,
    },
    "B": {
        "label": "4x4+2x2+1x1",
        "scales": [4, 2, 1],
        "tiles": 21,
    },
    "C": {
        "label": "4x4+5x5+2x2+1x1",
        "scales": [4, 5, 2, 1],
        "tiles": 46,
    },
    "D": {
        "label": "4x4+5x5",
        "scales": [4, 5],
        "tiles": 41,
    },
    "E": {
        "label": "4x4+5x5+6x6",
        "scales": [4, 5, 6],
        "tiles": 77,
    },
    "F": {
        "label": "4x4+6x6",
        "scales": [4, 6],
        "tiles": 52,
    },
    "G": {
        "label": "5x5",
        "scales": [5],
        "tiles": 25,
    },
    "H": {
        "label": "5x5+6x6",
        "scales": [5, 6],
        "tiles": 61,
    },
    "I": {
        "label": "6x6",
        "scales": [6],
        "tiles": 36,
    },
    "J": {
        "label": "4x4+5x5+6x6+3x3",
        "scales": [3, 4, 5, 6],
        "tiles": 86,
    },
    "K": {
        "label": "4x4+5x5+6x6+7x7",
        "scales": [4, 5, 6, 7],
        "tiles": 126,
    },
}

# Phase 1 sweep grid: (scheme_id, thresholds)
PHASE1_GRID = [
    ("A", [0.04]),
    ("B", [0.04, 0.06, 0.08, 0.10, 0.15]),
    ("C", [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]),
    ("D", [0.04, 0.06, 0.075, 0.08, 0.10, 0.15]),
    ("E", [0.04, 0.06, 0.075, 0.08, 0.10, 0.15]),
    ("F", [0.04, 0.06, 0.075, 0.08, 0.10, 0.15]),
    ("G", [0.04, 0.06, 0.075, 0.08, 0.10]),
    ("H", [0.04, 0.06, 0.075, 0.08, 0.10, 0.15]),
    ("I", [0.04, 0.06, 0.075, 0.08, 0.10]),
    ("J", [0.04, 0.06, 0.075, 0.08, 0.10, 0.15]),
    ("K", [0.04, 0.06, 0.075, 0.08, 0.10, 0.15]),
]


# --------------------------------------------------------------------------
# Multi-scale tile loading
# --------------------------------------------------------------------------

def load_scale_tiles(
    parquet_path: str,
    scale: int,
) -> pd.DataFrame:
    """Load per-tile data for a specific scale.

    Accepts either:
      - A direct parquet file path (e.g., /path/to/scale_2x2.parquet)
      - A directory with shard format (e.g., /path/to/shard_XXXX/*.parquet)
      - A directory containing scale-specific files (looks for scale_NxN.parquet)

    When given a directory with multiple parquet files, selects only the
    file matching scale_NxN.parquet to avoid loading other scales.
    """
    import glob as _glob

    if os.path.isfile(parquet_path):
        # Direct file path
        df = pd.read_parquet(parquet_path)
    elif os.path.isdir(parquet_path):
        # Check for scale-specific file first
        scale_file = os.path.join(parquet_path, f"scale_{scale}x{scale}.parquet")
        if os.path.exists(scale_file):
            df = pd.read_parquet(scale_file)
        else:
            # Fall back to shard format (Exp-3/Exp-4 style)
            shard_files = sorted(_glob.glob(
                os.path.join(parquet_path, "shard_*/*.parquet")
            ))
            if shard_files:
                df = load_per_tile_parquet(parquet_path, pattern="shard_*/*.parquet")
            else:
                raise FileNotFoundError(
                    f"No scale_{scale}x{scale}.parquet or shard_*/*.parquet "
                    f"found in {parquet_path}"
                )
    else:
        raise FileNotFoundError(f"Path does not exist: {parquet_path}")

    df["scale"] = scale
    # Reindex tile_index to be unique across scales: scale * 1000 + tile_index
    if "tile_index" in df.columns:
        df["tile_index"] = scale * 1000 + df["tile_index"]
    return df


def load_multiscale_tiles(
    scale_dirs: Dict[int, str],
) -> pd.DataFrame:
    """Load and combine per-tile data from multiple scale directories.

    Args:
        scale_dirs: {scale: parquet_root} mapping

    Returns:
        Combined DataFrame with all tiles tagged with 'scale' column.
    """
    frames = []
    for scale, path in sorted(scale_dirs.items()):
        print(f"  Loading scale {scale}x{scale} from {path}...")
        t0 = time.time()
        df = load_scale_tiles(path, scale)
        n_images = df["image_name"].nunique()
        expected_tiles_per_image = scale * scale
        actual_tiles_per_image = len(df) / n_images if n_images > 0 else 0
        print(f"    -> {len(df)} tiles, {n_images} images, "
              f"{actual_tiles_per_image:.1f} tiles/img "
              f"(expected {expected_tiles_per_image}), "
              f"{time.time() - t0:.1f}s")
        if abs(actual_tiles_per_image - expected_tiles_per_image) > 0.5:
            print(f"    *** WARNING: tile count mismatch! "
                  f"Expected {expected_tiles_per_image} tiles/img for scale {scale}x{scale}, "
                  f"got {actual_tiles_per_image:.1f}. Check data source.")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate_multiscale(
    tile_data: pd.DataFrame,
    config: AggregationConfig,
    image_col: str = "image_name",
    agg_method: str = "max_prob",
    agg_quantile: float = 0.9,
) -> Dict[str, np.ndarray]:
    """Aggregate multi-scale tiles per image using configurable method.

    Args:
        agg_method: 'max_prob', 'quantile', 'mean_prob', 'noisy_or', etc.
        agg_quantile: Quantile value when agg_method='quantile' (0.0-1.0).
    """
    aggregator = create_aggregator(
        method=agg_method,
        num_classes=config.num_classes,
        temperature=config.temperature,
        denoise_eps=config.denoise_eps,
        image_topk=config.image_topk if config.image_topk > 0 else None,
        output_dtype=np.float16,
        agg_quantile=agg_quantile,
    )

    geo_mask = None
    if config.geo_mask_path:
        geo_mask = np.load(config.geo_mask_path)
        if geo_mask.dtype != np.bool_:
            geo_mask = geo_mask.astype(np.bool_)

    results = {}
    for name, group in tile_data.groupby(image_col):
        # Aug-consistency filter (skip for new scales where all pass)
        if config.aug_filter and "passed_aug_consistency" in group.columns:
            group = group[group["passed_aug_consistency"] == True]  # noqa: E712

        if group.empty:
            continue

        # Tile confidence filter (k=0 or k>=num_tiles means keep all)
        if "max_prob" in group.columns:
            group = group.sort_values("max_prob", ascending=False)
        if config.tile_filter_k > 0 and len(group) > config.tile_filter_k:
            group = group.head(config.tile_filter_k)

        # Aggregate logits
        logits_bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, config.num_classes)
        p_image = aggregator.aggregate(tile_logits)

        # Geo mask
        if geo_mask is not None:
            p_f32 = p_image.astype(np.float32)
            p_f32[~geo_mask] = 0.0
            p_image = p_f32.astype(np.float16)

        results[str(name)] = p_image

    return results


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------

def run_sweep(
    all_scale_dirs: Dict[int, str],
    model_path: str,
    species_map_path: str,
    class_mask_path: Optional[str],
    out_dir: str,
    temperature: float = 1.5,
    alpha: float = 0.6,
    topk: int = 10,
    image_topk: int = 64,
    tile_filter_k: int = 0,  # 0 = keep ALL tiles
    sweep_grid: Optional[List[Tuple[str, List[float]]]] = None,
    agg_methods: Optional[List[str]] = None,
    agg_quantiles: Optional[List[float]] = None,
) -> None:
    """Run the aggregation method sweep.

    For each (scheme, agg_method, quantile) combo, aggregates tiles,
    applies temporal fusion, and generates submission CSVs at each threshold.
    """

    if sweep_grid is None:
        sweep_grid = PHASE1_GRID
    if agg_methods is None:
        agg_methods = ["max_prob"]
    if agg_quantiles is None:
        agg_quantiles = [0.90]

    # Build the full list of (method, quantile) combos to sweep
    method_configs: List[Tuple[str, float]] = []
    for method in agg_methods:
        if method == "quantile":
            for q in agg_quantiles:
                method_configs.append(("quantile", q))
        else:
            method_configs.append((method, 0.0))  # quantile unused

    os.makedirs(out_dir, exist_ok=True)

    species_map = load_species_map(species_map_path, NUM_CLASSES)
    class_mask = (
        load_class_mask_npy(class_mask_path, NUM_CLASSES)
        if class_mask_path
        else None
    )

    # Pre-load all scale data
    print("=" * 70)
    print("Loading tile data from all scales...")
    print("=" * 70)
    scale_data: Dict[int, pd.DataFrame] = {}
    for scale, path in all_scale_dirs.items():
        print(f"  Scale {scale}x{scale}: {path}")
        t0 = time.time()
        df = load_scale_tiles(path, scale)
        scale_data[scale] = df
        n_images = df["image_name"].nunique()
        print(f"    -> {len(df)} tiles, {n_images} images, {time.time() - t0:.1f}s")

    results_summary = []

    for scheme_id, thresholds in sweep_grid:
        scheme = SCHEMES[scheme_id]
        scheme_label = scheme["label"]
        required_scales = scheme["scales"]

        # Check all required scales are available
        missing = [s for s in required_scales if s not in scale_data]
        if missing:
            print(f"\n*** Skipping scheme {scheme_id} ({scheme_label}): "
                  f"missing scales {missing}")
            continue

        # Combine tiles from relevant scales (shared across all methods)
        frames = [scale_data[s] for s in required_scales]
        combined = pd.concat(frames, ignore_index=True)
        n_images = combined["image_name"].nunique()

        for method, quantile in method_configs:
            method_label = method if method != "quantile" else f"quantile_q{quantile:.2f}"

            print(f"\n{'=' * 70}")
            print(f"Scheme {scheme_id}: {scheme_label} ({scheme['tiles']} tiles) | "
                  f"Agg: {method_label}")
            print(f"{'=' * 70}")
            print(f"Combined: {len(combined)} tiles, {n_images} images")

            # Aggregate
            config = AggregationConfig(
                temperature=temperature,
                aug_filter=True,
                tile_filter_k=tile_filter_k,
                geo_mask_path=class_mask_path,
                image_topk=image_topk,
                num_classes=NUM_CLASSES,
            )

            print(f"Aggregating ({method_label}, T={temperature}, "
                  f"k={'ALL' if tile_filter_k <= 0 else tile_filter_k})...")
            t0 = time.time()
            image_probs = aggregate_multiscale(
                combined, config,
                agg_method=method,
                agg_quantile=quantile,
            )
            print(f"  -> {len(image_probs)} images in {time.time() - t0:.1f}s")

            # Save pre-fusion parquet
            tag = f"scheme_{scheme_id}_{method_label}_T{temperature:.2f}"
            prefusion_path = os.path.join(out_dir, f"{tag}_prefusion.parquet")
            save_aggregated(image_probs, prefusion_path)

            # Temporal fusion (prior_blend, alpha)
            print(f"Temporal fusion (prior_blend, alpha={alpha})...")
            t0 = time.time()
            quadrat_probs = load_quadrat_probs(prefusion_path)
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
            print(f"  -> Fused {len(fused_probs)} quadrats in {time.time() - t0:.1f}s")

            # Save fused parquet
            fused_f16 = {
                qid: p.astype(np.float16) for qid, p in fused_probs.items()
            }
            fused_path = os.path.join(out_dir, f"{tag}_fused_a{alpha}.parquet")
            save_aggregated(fused_f16, fused_path)

            # Generate CSVs at each threshold
            for thr in thresholds:
                csv_name = f"{tag}_a{alpha}_thr{thr:.3f}_top{topk}.csv"
                csv_path = os.path.join(out_dir, csv_name)

                n_rows = generate_submission_csv(
                    in_parquet=fused_path,
                    out_csv=csv_path,
                    species_map=species_map,
                    num_classes=NUM_CLASSES,
                    class_mask=class_mask,
                    topk=topk,
                    threshold=thr,
                    min_k=1,
                    id_col="image_name",
                )

                results_summary.append({
                    "run": len(results_summary) + 1,
                    "scheme": scheme_id,
                    "label": scheme_label,
                    "tiles": scheme["tiles"],
                    "agg": method_label,
                    "threshold": thr,
                    "images": n_rows,
                    "csv": csv_name,
                })

    # Print summary
    print(f"\n{'=' * 100}")
    print("EXP-8 AGGREGATION SWEEP SUMMARY")
    print(f"{'=' * 100}")
    print(f"Temperature: {temperature}  Alpha: {alpha}  TopK: {topk}  "
          f"tile_filter_k: {'ALL' if tile_filter_k <= 0 else tile_filter_k}")
    print(f"Methods: {[m for m, _ in method_configs]}")
    print()

    fmt = "{:>4} | {:>7} | {:>20} | {:>5} | {:>18} | {:>9} | {:>6} | {}"
    print(fmt.format("Run", "Scheme", "Label", "Tiles", "Aggregation", "Threshold", "Images", "CSV"))
    print(fmt.format("-" * 4, "-" * 7, "-" * 20, "-" * 5, "-" * 18, "-" * 9, "-" * 6, "-" * 40))
    for r in results_summary:
        print(fmt.format(
            r["run"], r["scheme"], r["label"], r["tiles"], r["agg"],
            f"{r['threshold']:.3f}", r["images"], r["csv"],
        ))
    print(f"{'=' * 100}")
    print(f"\nTotal CSVs: {len(results_summary)}")
    print(f"Output: {out_dir}")
    print("\nSubmit CSVs to Kaggle and record F1 scores.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp-8: Quantile aggregation sweep on multi-scale tiles"
    )

    # Scale directories
    ap.add_argument("--scale4-dir", required=True,
                    help="Parquet root for 4x4 tiles (existing JPEG85 embeddings)")
    ap.add_argument("--scale1-dir", default=None,
                    help="Parquet root for 1x1 tiles")
    ap.add_argument("--scale2-dir", default=None,
                    help="Parquet root for 2x2 tiles")
    ap.add_argument("--scale5-dir", default=None,
                    help="Parquet root for 5x5 tiles")
    ap.add_argument("--scale6-dir", default=None,
                    help="Parquet root for 6x6 tiles")
    ap.add_argument("--scale3-dir", default=None,
                    help="Parquet root for 3x3 tiles")
    ap.add_argument("--scale7-dir", default=None,
                    help="Parquet root for 7x7 tiles")

    # Model and data
    ap.add_argument("--model-path", required=True,
                    help="DINOv2 checkpoint (.pth) — needed for temporal fusion")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None,
                    help="Geo-mask (sw_europe_mask_bool.npy)")
    ap.add_argument("--out-dir", required=True)

    # Aggregation params
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Temporal fusion alpha (prior_blend)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--tile-filter-k", type=int, default=0,
                    help="Tile filter k (0=ALL, disabled)")

    # Sweep customization
    ap.add_argument("--schemes", default="E",
                    help="Comma-separated scheme IDs to test (default: E)")
    ap.add_argument("--thresholds", default="0.06,0.075,0.08,0.10",
                    help="Override thresholds (comma-separated)")

    # Aggregation method sweep (NEW in Exp-8)
    ap.add_argument("--agg-methods", default="max_prob,quantile",
                    help="Comma-separated aggregation methods to sweep "
                         "(default: max_prob,quantile)")
    ap.add_argument("--agg-quantiles", default="0.75,0.80,0.85,0.90,0.95",
                    help="Comma-separated quantile values for quantile method "
                         "(default: 0.75,0.80,0.85,0.90,0.95)")

    args = ap.parse_args()

    # Build scale directory mapping
    all_scale_dirs: Dict[int, str] = {}
    all_scale_dirs[4] = args.scale4_dir
    if args.scale1_dir:
        all_scale_dirs[1] = args.scale1_dir
    if args.scale2_dir:
        all_scale_dirs[2] = args.scale2_dir
    if args.scale5_dir:
        all_scale_dirs[5] = args.scale5_dir
    if args.scale6_dir:
        all_scale_dirs[6] = args.scale6_dir
    if args.scale3_dir:
        all_scale_dirs[3] = args.scale3_dir
    if args.scale7_dir:
        all_scale_dirs[7] = args.scale7_dir

    # Build sweep grid
    requested_schemes = [s.strip() for s in args.schemes.split(",")]
    custom_thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    sweep_grid = [(s, custom_thresholds) for s in requested_schemes]

    # Parse aggregation methods
    agg_methods = [m.strip() for m in args.agg_methods.split(",")]
    agg_quantiles = [float(q.strip()) for q in args.agg_quantiles.split(",")]

    print(f"Exp-8: Quantile Aggregation Sweep")
    print(f"Scales available: {sorted(all_scale_dirs.keys())}")
    print(f"Schemes: {requested_schemes}")
    print(f"Aggregation methods: {agg_methods}")
    if "quantile" in agg_methods:
        print(f"Quantile values: {agg_quantiles}")
    print()

    run_sweep(
        all_scale_dirs=all_scale_dirs,
        model_path=args.model_path,
        species_map_path=args.species_map,
        class_mask_path=args.class_mask_npy,
        out_dir=args.out_dir,
        temperature=args.temperature,
        alpha=args.alpha,
        topk=args.topk,
        image_topk=args.image_topk,
        tile_filter_k=args.tile_filter_k,
        sweep_grid=sweep_grid,
        agg_methods=agg_methods,
        agg_quantiles=agg_quantiles,
    )


if __name__ == "__main__":
    main()
