"""Step 6: Six-view aggregation sweep.

Extends the 3-view per_view_or strategy (F1=0.315) to 6 views by adding
rot180, rot270, and vflip augmentation embeddings extracted in the previous step.

More views = lower variance in per_view_or averaging, potentially more robust predictions.
Compares 3-view baseline vs 6-view to isolate the effect of additional augmentations.

Views:
  3-view (baseline): orig, hflip, rot90
  6-view (new):      orig, hflip, rot90, rot180, rot270, vflip

Sweep: {3-view, 6-view} x max_prob x T=1.5 x k={3,4,5,6}
"""
from __future__ import annotations

import argparse
import os
import time

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
from multiview_aggregate import (
    EMB_DIM,
    NUM_CLASSES,
    compute_logits_numpy,
    decode_embeddings,
    load_classifier_head,
    _filter_tiles,
)
from utils.aggregator import create_aggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix


# View definitions: (column_name, source_type)
# "logits" = stored directly as logits in parquet
# "embedding" = stored as CLS embedding, compute logits via matmul
VIEW_SPECS = {
    "orig":  ("teacher_logits_f16", "logits"),
    "hflip": ("embedding_flip_f16", "embedding"),
    "rot90": ("embedding_rot90_f16", "embedding"),
    "rot180": ("embedding_rot180_f16", "embedding"),
    "rot270": ("embedding_rot270_f16", "embedding"),
    "vflip": ("embedding_vflip_f16", "embedding"),
}

VIEWS_3 = ["orig", "hflip", "rot90"]
VIEWS_6 = ["orig", "hflip", "rot90", "rot180", "rot270", "vflip"]


def load_additional_views_parquet(parquet_path: str) -> pd.DataFrame:
    """Load the supplementary parquet with rot180/rot270/vflip embeddings."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    return table.to_pandas()


def merge_tile_data(
    original: pd.DataFrame,
    additional: pd.DataFrame,
) -> pd.DataFrame:
    """Merge original Phase 1 tile data with additional view embeddings."""
    merged = original.merge(
        additional,
        on=["image_name", "tile_index"],
        how="inner",
    )
    n_orig = len(original)
    n_merged = len(merged)
    if n_merged < n_orig:
        print(
            f"  Warning: {n_orig - n_merged} tiles dropped during merge "
            f"(original={n_orig}, merged={n_merged})"
        )
    return merged


def get_view_logits(
    group: pd.DataFrame,
    view_name: str,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
) -> np.ndarray:
    """Get logits for a view, either from stored logits or by computing from embeddings."""
    col_name, source_type = VIEW_SPECS[view_name]

    if source_type == "logits":
        return f16_bytes_list_to_f32_matrix(
            group[col_name].tolist(), NUM_CLASSES
        )
    else:
        embeddings = decode_embeddings(group[col_name].tolist())
        return compute_logits_numpy(embeddings, head_weight, head_bias)


def nview_per_view_aggregate(
    tile_data: pd.DataFrame,
    views: list,
    agg_method: str,
    config: AggregationConfig,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    image_col: str = "image_name",
):
    """Run per_view_or aggregation with N views. Returns {image_name: probs_f16}."""
    image_topk = config.image_topk if config.image_topk > 0 else None
    n_views = len(views)

    # Aggregate each view WITHOUT image_topk, then average, then apply image_topk
    aggregator = create_aggregator(
        method=agg_method,
        num_classes=config.num_classes,
        temperature=config.temperature,
        denoise_eps=config.denoise_eps,
        image_topk=None,
        output_dtype=np.float32,
    )

    geo_mask = None
    if config.geo_mask_path:
        geo_mask = np.load(config.geo_mask_path)
        if geo_mask.dtype != np.bool_:
            geo_mask = geo_mask.astype(np.bool_)

    results = {}
    for name, group in tile_data.groupby(image_col):
        group = _filter_tiles(group, config)
        if group.empty:
            continue

        # Aggregate each view independently
        view_probs = []
        for view_name in views:
            logits = get_view_logits(group, view_name, head_weight, head_bias)
            p_view = aggregator.aggregate(logits).astype(np.float32)
            view_probs.append(p_view)

        # Average across views
        p_avg = np.mean(view_probs, axis=0)

        # Apply image_topk after averaging
        if image_topk is not None:
            k = min(image_topk, p_avg.shape[0])
            if k < p_avg.shape[0]:
                idx = np.argpartition(p_avg, -k)[-k:]
                mask = np.zeros(p_avg.shape[0], dtype=np.bool_)
                mask[idx] = True
                p_avg[~mask] = 0.0

        # Geo mask
        if geo_mask is not None:
            p_avg[~geo_mask] = 0.0

        results[str(name)] = p_avg.astype(np.float16)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Six-view aggregation sweep (3-view vs 6-view per_view_or)"
    )

    ap.add_argument("--tile-parquet-root", required=True)
    ap.add_argument(
        "--additional-views-parquet", required=True,
        help="Parquet with rot180/rot270/vflip embeddings from extract_additional_views.py",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--model-path", required=True,
        help="DINOv2 checkpoint (.pth) for extracting classifier head",
    )
    ap.add_argument(
        "--model-name", default="vit_base_patch14_reg4_dinov2.lvd142m"
    )

    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)

    ap.add_argument(
        "--agg-method", default="max_prob",
        help="Aggregation method (default: max_prob, the Step 4 winner)",
    )
    ap.add_argument(
        "--temperatures", default="1.5",
        help="Comma-separated temperatures",
    )
    ap.add_argument(
        "--tile-filter-ks", default="3,4,5,6",
        help="Comma-separated tile-filter-k values",
    )
    ap.add_argument("--thresholds", default="0.03")
    ap.add_argument("--topk", type=int, default=10)

    ap.add_argument("--aug-filter", action="store_true", default=True)
    ap.add_argument("--no-aug-filter", action="store_false", dest="aug_filter")
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--id-col", default="image_name")

    args = ap.parse_args()

    temps = [float(t.strip()) for t in args.temperatures.split(",")]
    tile_ks = [int(k.strip()) for k in args.tile_filter_ks.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)

    # Load classifier head
    print(f"Loading classifier head from {args.model_path}...")
    t0 = time.time()
    head_weight, head_bias = load_classifier_head(args.model_path, args.model_name)
    print(f"Head loaded in {time.time() - t0:.1f}s  weight={head_weight.shape}")

    # Load original tile data
    print(f"Loading original tile data from {args.tile_parquet_root}...")
    tile_data_orig = load_per_tile_parquet(args.tile_parquet_root)
    print(f"Loaded {len(tile_data_orig)} tile rows (original)")

    # Load additional views
    print(f"Loading additional views from {args.additional_views_parquet}...")
    additional_views = load_additional_views_parquet(args.additional_views_parquet)
    print(f"Loaded {len(additional_views)} tile rows (additional views)")

    # Merge
    print("Merging original + additional view data...")
    tile_data_merged = merge_tile_data(tile_data_orig, additional_views)
    print(f"Merged: {len(tile_data_merged)} tile rows")

    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    # Sweep configs: 3-view (baseline) and 6-view (new)
    view_configs = [
        ("3view", VIEWS_3, tile_data_orig),
        ("6view", VIEWS_6, tile_data_merged),
    ]

    n_combos = len(view_configs) * len(temps) * len(tile_ks) * len(thresholds)
    print(
        f"\nSweep: {len(view_configs)} view-configs x {len(temps)} temps "
        f"x {len(tile_ks)} tile-ks x {len(thresholds)} thresholds "
        f"= {n_combos} submissions"
    )

    results_summary = []

    for view_label, views, data in view_configs:
        for temp in temps:
            for tile_k in tile_ks:
                print(f"\n{'='*70}")
                print(
                    f"{view_label} per_view_or + {args.agg_method}  "
                    f"T={temp}  k={tile_k}  views={len(views)}"
                )
                print(f"{'='*70}")

                config = AggregationConfig(
                    temperature=temp,
                    aug_filter=args.aug_filter,
                    tile_filter_k=tile_k,
                    geo_mask_path=args.class_mask_npy,
                    image_topk=args.image_topk,
                    num_classes=args.num_classes,
                )

                t0 = time.time()
                image_probs = nview_per_view_aggregate(
                    data,
                    views=views,
                    agg_method=args.agg_method,
                    config=config,
                    head_weight=head_weight,
                    head_bias=head_bias,
                    image_col=args.id_col,
                )
                elapsed = time.time() - t0
                print(f"Aggregated {len(image_probs)} images in {elapsed:.1f}s")

                tag = (
                    f"mv_{view_label}_per_view_or_{args.agg_method}"
                    f"_T{temp:.1f}_k{tile_k}"
                )
                parquet_path = os.path.join(args.out_dir, f"{tag}.parquet")
                save_aggregated(
                    image_probs, parquet_path, id_col=args.id_col
                )

                for thr in thresholds:
                    csv_path = os.path.join(
                        args.out_dir,
                        f"{tag}_thr{thr:.2f}_top{args.topk}.csv",
                    )
                    n_rows = generate_submission_csv(
                        in_parquet=parquet_path,
                        out_csv=csv_path,
                        species_map=species_map,
                        num_classes=args.num_classes,
                        class_mask=class_mask,
                        topk=args.topk,
                        threshold=thr,
                        min_k=1,
                        id_col=args.id_col,
                    )
                    results_summary.append(
                        (view_label, len(views), temp, tile_k, thr, n_rows, csv_path)
                    )

    # Print summary
    print(f"\n{'='*90}")
    print("SIX-VIEW SWEEP SUMMARY")
    print(f"{'='*90}")
    fmt = "{:>8} {:>6} {:>5} {:>6} {:>6} {:>8}  {}"
    print(fmt.format("Views", "NView", "Temp", "TileK", "Thr", "Images", "CSV"))
    print(
        fmt.format(
            "-" * 8, "-" * 6, "-" * 5, "-" * 6, "-" * 6, "-" * 8, "-" * 50
        )
    )
    for label, nv, temp, tile_k, thr, n, path in results_summary:
        print(
            fmt.format(
                label, str(nv), f"{temp:.1f}", str(tile_k),
                f"{thr:.2f}", str(n), path,
            )
        )
    print(f"{'='*90}")
    print(f"\nBaseline to beat: 3-view per_view_or + max_prob T1.5 k3 = F1 0.31487")


if __name__ == "__main__":
    main()
