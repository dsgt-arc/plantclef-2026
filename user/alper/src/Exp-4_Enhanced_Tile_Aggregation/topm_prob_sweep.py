"""Step 5: Top-m probability aggregation sweep.

Tests the top_m_prob aggregator (average of top-m tile probabilities per class)
with the per_view_or strategy that won Step 4.

This is a natural extension of max_prob (m=1, F1=0.315): instead of relying on
a single tile, average the m most confident tiles per species. m=2 or m=3 may
be more robust while still ignoring noisy tiles.

Sweep: m={1,2,3} x T=1.5 x k={3,4,5,6} x per_view_or
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

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


def topm_per_view_aggregate(
    tile_data,
    agg_method: str,
    agg_m: int,
    config: AggregationConfig,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    image_col: str = "image_name",
):
    """Run per_view_or aggregation with top_m_prob. Returns {image_name: probs_f16}."""
    image_topk = config.image_topk if config.image_topk > 0 else None

    # Aggregate each view WITHOUT image_topk, then average, then apply image_topk
    aggregator = create_aggregator(
        method=agg_method,
        num_classes=config.num_classes,
        temperature=config.temperature,
        denoise_eps=config.denoise_eps,
        image_topk=None,
        output_dtype=np.float32,
        agg_m=agg_m,
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

        logits_orig = f16_bytes_list_to_f32_matrix(
            group["teacher_logits_f16"].tolist(), NUM_CLASSES
        )
        emb_flip = decode_embeddings(group["embedding_flip_f16"].tolist())
        emb_rot90 = decode_embeddings(group["embedding_rot90_f16"].tolist())
        logits_flip = compute_logits_numpy(emb_flip, head_weight, head_bias)
        logits_rot90 = compute_logits_numpy(emb_rot90, head_weight, head_bias)

        p_orig = aggregator.aggregate(logits_orig).astype(np.float32)
        p_flip = aggregator.aggregate(logits_flip).astype(np.float32)
        p_rot90 = aggregator.aggregate(logits_rot90).astype(np.float32)
        p_avg = (p_orig + p_flip + p_rot90) / 3.0

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
        description="Top-m probability aggregation sweep with per_view_or"
    )

    ap.add_argument("--tile-parquet-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--model-path",
        required=True,
        help="DINOv2 checkpoint (.pth) for extracting classifier head",
    )
    ap.add_argument(
        "--model-name", default="vit_base_patch14_reg4_dinov2.lvd142m"
    )

    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)

    ap.add_argument(
        "--agg-ms",
        default="1,2,3",
        help="Comma-separated m values for top-m averaging (default: 1,2,3)",
    )
    ap.add_argument(
        "--temperatures", default="1.5", help="Comma-separated temperatures"
    )
    ap.add_argument(
        "--tile-filter-ks",
        default="3,4,5,6",
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

    agg_ms = [int(m.strip()) for m in args.agg_ms.split(",")]
    temps = [float(t.strip()) for t in args.temperatures.split(",")]
    tile_ks = [int(k.strip()) for k in args.tile_filter_ks.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)

    # Load classifier head
    print(f"Loading classifier head from {args.model_path}...")
    t0 = time.time()
    head_weight, head_bias = load_classifier_head(args.model_path, args.model_name)
    print(f"Head loaded in {time.time() - t0:.1f}s  weight={head_weight.shape}")

    # Load tile data once
    print(f"Loading per-tile data from {args.tile_parquet_root}...")
    tile_data = load_per_tile_parquet(args.tile_parquet_root)
    print(f"Loaded {len(tile_data)} tile rows")

    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    n_combos = len(agg_ms) * len(temps) * len(tile_ks) * len(thresholds)
    print(
        f"Sweep: {len(agg_ms)} m-values x {len(temps)} temps "
        f"x {len(tile_ks)} tile-ks x {len(thresholds)} thresholds "
        f"= {n_combos} submissions"
    )

    results_summary = []

    for m in agg_ms:
        for temp in temps:
            for tile_k in tile_ks:
                print(f"\n{'='*70}")
                print(f"per_view_or + top{m}_prob  T={temp}  k={tile_k}")
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
                image_probs = topm_per_view_aggregate(
                    tile_data,
                    agg_method="top_m_prob",
                    agg_m=m,
                    config=config,
                    head_weight=head_weight,
                    head_bias=head_bias,
                    image_col=args.id_col,
                )
                elapsed = time.time() - t0
                print(f"Aggregated {len(image_probs)} images in {elapsed:.1f}s")

                tag = f"mv_per_view_or_top{m}_prob_T{temp:.1f}_k{tile_k}"
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
                    results_summary.append((m, temp, tile_k, thr, n_rows, csv_path))

    # Print summary
    print(f"\n{'='*90}")
    print("TOP-M PROBABILITY SWEEP SUMMARY")
    print(f"{'='*90}")
    fmt = "{:>4} {:>5} {:>6} {:>6} {:>8}  {}"
    print(fmt.format("M", "Temp", "TileK", "Thr", "Images", "CSV"))
    print(fmt.format("-" * 4, "-" * 5, "-" * 6, "-" * 6, "-" * 8, "-" * 50))
    for m, temp, tile_k, thr, n, path in results_summary:
        print(fmt.format(str(m), f"{temp:.1f}", str(tile_k), f"{thr:.2f}", str(n), path))
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
