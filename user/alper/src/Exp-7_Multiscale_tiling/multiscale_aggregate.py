"""Multi-scale tile aggregation sweep (Exp-7 Phase 1).

Loads per-tile logits from multiple scale directories, combines them per image,
runs max_prob aggregation, temporal fusion, and generates submission CSVs.

Three schemes tested:
  A (baseline): 4x4 only             (16 tiles)
  B:            4x4 + 2x2 + 1x1      (21 tiles)
  C:            4x4 + 5x5 + 2x2 + 1  (46 tiles)

Pipeline per scheme:
  1. Load + combine tiles from relevant scale directories
  2. max_prob aggregation (T=1.5, k=ALL, image_topk=64)
  3. Temporal fusion (prior_blend, alpha=0.6)
  4. Generate CSVs at multiple thresholds

Usage:
  python multiscale_aggregate.py \
    --scale4-dir /path/to/phase1_tiles_test_jpeg85 \
    --scale1-dir /path/to/scale_1x1 \
    --scale2-dir /path/to/scale_2x2 \
    --scale5-dir /path/to/scale_5x5 \
    --model-path /path/to/vit_base_patch14_reg4_dinov2.pth \
    --species-map /path/to/species_map.json \
    --class-mask-npy /path/to/sw_europe_mask_bool.npy \
    --out-dir /path/to/output
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
}

# Phase 1 sweep grid: (scheme_id, thresholds)
PHASE1_GRID = [
    ("A", [0.04]),
    ("B", [0.04, 0.06, 0.08, 0.10, 0.15]),
    ("C", [0.04, 0.06, 0.08, 0.10, 0.15, 0.20]),
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
) -> Dict[str, np.ndarray]:
    """Aggregate multi-scale tiles per image using max_prob.

    Like aggregate_tiles.run_aggregation but uses create_aggregator factory
    to support max_prob method.
    """
    aggregator = create_aggregator(
        method="max_prob",
        num_classes=config.num_classes,
        temperature=config.temperature,
        denoise_eps=config.denoise_eps,
        image_topk=config.image_topk if config.image_topk > 0 else None,
        output_dtype=np.float16,
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
) -> None:
    """Run the Phase 1 multi-scale sweep."""

    if sweep_grid is None:
        sweep_grid = PHASE1_GRID

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

        print(f"\n{'=' * 70}")
        print(f"Scheme {scheme_id}: {scheme_label} ({scheme['tiles']} tiles)")
        print(f"{'=' * 70}")

        # Combine tiles from relevant scales
        frames = [scale_data[s] for s in required_scales]
        combined = pd.concat(frames, ignore_index=True)
        n_images = combined["image_name"].nunique()
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

        print(f"Aggregating (max_prob, T={temperature}, k={'ALL' if tile_filter_k <= 0 else tile_filter_k})...")
        t0 = time.time()
        image_probs = aggregate_multiscale(combined, config)
        print(f"  -> {len(image_probs)} images in {time.time() - t0:.1f}s")

        # Save pre-fusion parquet
        tag = f"scheme_{scheme_id}"
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
            csv_name = f"{tag}_thr{thr:.2f}_top{topk}.csv"
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
                "threshold": thr,
                "images": n_rows,
                "csv": csv_name,
            })

    # Print summary
    print(f"\n{'=' * 90}")
    print("PHASE 1 SWEEP SUMMARY")
    print(f"{'=' * 90}")
    print(f"Temperature: {temperature}  Alpha: {alpha}  TopK: {topk}  "
          f"tile_filter_k: {'ALL' if tile_filter_k <= 0 else tile_filter_k}")
    print()

    fmt = "{:>4} | {:>7} | {:>25} | {:>5} | {:>9} | {:>6} | {}"
    print(fmt.format("Run", "Scheme", "Label", "Tiles", "Threshold", "Images", "CSV"))
    print(fmt.format("-" * 4, "-" * 7, "-" * 25, "-" * 5, "-" * 9, "-" * 6, "-" * 40))
    for r in results_summary:
        print(fmt.format(
            r["run"], r["scheme"], r["label"], r["tiles"],
            f"{r['threshold']:.2f}", r["images"], r["csv"],
        ))
    print(f"{'=' * 90}")
    print(f"\nTotal CSVs: {len(results_summary)}")
    print(f"Output: {out_dir}")
    print("\nSubmit CSVs to Kaggle and record F1 scores.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp-7 Phase 1: Multi-scale tile aggregation sweep"
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
    ap.add_argument("--schemes", default="A,B,C",
                    help="Comma-separated scheme IDs to test (default: A,B,C)")
    ap.add_argument("--thresholds", default=None,
                    help="Override thresholds (comma-separated). "
                         "If set, same thresholds for all schemes.")

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

    # Build sweep grid
    requested_schemes = [s.strip() for s in args.schemes.split(",")]
    if args.thresholds:
        custom_thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        sweep_grid = [(s, custom_thresholds) for s in requested_schemes]
    else:
        sweep_grid = [(s, thrs) for s, thrs in PHASE1_GRID if s in requested_schemes]

    print(f"Exp-7 Phase 1: Multi-Scale Tiling Sweep")
    print(f"Scales available: {sorted(all_scale_dirs.keys())}")
    print(f"Schemes: {requested_schemes}")
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
    )


if __name__ == "__main__":
    main()
