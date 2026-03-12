"""Multi-view tile aggregation: compute logits from stored flip/rot90 embeddings,
combine views, then aggregate across tiles. No re-extraction needed — uses
embeddings already stored in Phase 1 parquet.

View-combination strategies:
  avg_logits  — Average orig/flip/rot90 logits per tile, then aggregate across tiles
  all_tiles   — Treat each view as a separate tile (3x tiles in aggregator)
  per_view_or — Aggregate each view separately, average 3 image-level prob vectors
  orig_only   — Baseline (only orig logits, for comparison)

Aggregation methods:
  noisy_or          — Standard Noisy-OR (default)
  weighted_noisy_or — Per-tile confidence-weighted Noisy-OR
  mean_prob         — Simple mean of per-tile softmax probs
  max_prob          — Max per-tile softmax prob per class
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import timm

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
from utils.aggregator import create_aggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes

EMB_DIM = 768
NUM_CLASSES = 7806

VALID_STRATEGIES = ("avg_logits", "all_tiles", "per_view_or", "orig_only")
VALID_AGG_METHODS = ("noisy_or", "weighted_noisy_or", "mean_prob", "max_prob")


def load_classifier_head(
    model_path: str,
    model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
) -> tuple:
    """Load DINOv2 model and extract classifier head weight/bias as numpy arrays.

    Returns (weight [7806, 768], bias [7806]).
    """
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
        checkpoint_path=model_path,
    )
    head = model.head
    weight = head.weight.detach().cpu().numpy().astype(np.float32)  # [7806, 768]
    bias = head.bias.detach().cpu().numpy().astype(np.float32)  # [7806]
    del model
    return weight, bias


def decode_embeddings(bufs: list, dim: int = EMB_DIM) -> np.ndarray:
    """Decode f16 byte buffers to f32 matrix [N, dim]."""
    n = len(bufs)
    out = np.empty((n, dim), dtype=np.float32)
    for i, b in enumerate(bufs):
        v = np.frombuffer(b, dtype=np.float16)
        if v.size != dim:
            raise ValueError(f"Expected {dim} f16 values, got {v.size}")
        out[i] = v.astype(np.float32)
    return out


def compute_logits_numpy(
    embeddings_f32: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """embeddings [N, 768] @ weight.T [768, 7806] + bias -> logits [N, 7806]"""
    return embeddings_f32 @ weight.T + bias


def _filter_tiles(group, config: AggregationConfig):
    """Apply aug-consistency filter and top-k tile filter."""
    if config.aug_filter and "passed_aug_consistency" in group.columns:
        group = group[group["passed_aug_consistency"] == True]  # noqa: E712
    if group.empty:
        return group
    if "max_prob" in group.columns:
        group = group.sort_values("max_prob", ascending=False)
    if config.tile_filter_k > 0 and len(group) > config.tile_filter_k:
        group = group.head(config.tile_filter_k)
    return group


def multiview_aggregate(
    tile_data,
    strategy: str,
    agg_method: str,
    config: AggregationConfig,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    image_col: str = "image_name",
):
    """Run multi-view aggregation. Returns {image_name: probs_f16}."""
    aggregator = create_aggregator(
        method=agg_method,
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
        group = _filter_tiles(group, config)
        if group.empty:
            continue

        # Original logits (stored in parquet as 'teacher_logits_f16' — legacy column name)
        logits_orig = f16_bytes_list_to_f32_matrix(
            group["teacher_logits_f16"].tolist(), NUM_CLASSES
        )

        if strategy == "orig_only":
            p_image = aggregator.aggregate(logits_orig)
        else:
            # Compute flip/rot90 logits from stored embeddings
            emb_flip = decode_embeddings(group["embedding_flip_f16"].tolist())
            emb_rot90 = decode_embeddings(group["embedding_rot90_f16"].tolist())
            logits_flip = compute_logits_numpy(emb_flip, head_weight, head_bias)
            logits_rot90 = compute_logits_numpy(emb_rot90, head_weight, head_bias)

            if strategy == "avg_logits":
                tile_logits = (logits_orig + logits_flip + logits_rot90) / 3.0
                p_image = aggregator.aggregate(tile_logits)
            elif strategy == "all_tiles":
                tile_logits = np.concatenate(
                    [logits_orig, logits_flip, logits_rot90], axis=0
                )
                p_image = aggregator.aggregate(tile_logits)
            elif strategy == "per_view_or":
                p_orig = aggregator.aggregate(logits_orig).astype(np.float32)
                p_flip = aggregator.aggregate(logits_flip).astype(np.float32)
                p_rot90 = aggregator.aggregate(logits_rot90).astype(np.float32)
                p_image = ((p_orig + p_flip + p_rot90) / 3.0).astype(np.float16)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        # Geo mask
        if geo_mask is not None:
            p_f32 = p_image.astype(np.float32)
            p_f32[~geo_mask] = 0.0
            p_image = p_f32.astype(np.float16)

        results[str(name)] = p_image

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-view tile aggregation sweep"
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
        "--strategies",
        default="avg_logits,all_tiles,per_view_or,orig_only",
        help=f"Comma-separated view strategies: {VALID_STRATEGIES}",
    )
    ap.add_argument(
        "--aggregation-methods",
        default="noisy_or",
        help=f"Comma-separated aggregation methods: {VALID_AGG_METHODS}",
    )
    ap.add_argument(
        "--temperatures", default="2.0", help="Comma-separated temperatures"
    )
    ap.add_argument(
        "--tile-filter-ks",
        default="6",
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

    strategies = [s.strip() for s in args.strategies.split(",")]
    agg_methods = [m.strip() for m in args.aggregation_methods.split(",")]
    temps = [float(t.strip()) for t in args.temperatures.split(",")]
    tile_ks = [int(k.strip()) for k in args.tile_filter_ks.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    for s in strategies:
        if s not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{s}'. Valid: {VALID_STRATEGIES}")
    for m in agg_methods:
        if m not in VALID_AGG_METHODS:
            raise ValueError(f"Unknown method '{m}'. Valid: {VALID_AGG_METHODS}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load classifier head (numpy matmul, no GPU needed)
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

    n_combos = (
        len(strategies)
        * len(agg_methods)
        * len(temps)
        * len(tile_ks)
        * len(thresholds)
    )
    print(
        f"Sweep: {len(strategies)} strategies x {len(agg_methods)} agg_methods "
        f"x {len(temps)} temps x {len(tile_ks)} tile-ks "
        f"x {len(thresholds)} thresholds = {n_combos} submissions"
    )

    results_summary = []

    for strategy in strategies:
        for agg_method in agg_methods:
            for temp in temps:
                for tile_k in tile_ks:
                    print(f"\n{'='*70}")
                    print(
                        f"strategy={strategy}  agg={agg_method}  "
                        f"T={temp}  k={tile_k}"
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
                    image_probs = multiview_aggregate(
                        tile_data,
                        strategy,
                        agg_method,
                        config,
                        head_weight,
                        head_bias,
                        image_col=args.id_col,
                    )
                    elapsed = time.time() - t0
                    print(
                        f"Aggregated {len(image_probs)} images in {elapsed:.1f}s"
                    )

                    tag = f"mv_{strategy}_{agg_method}_T{temp:.1f}_k{tile_k}"
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
                            (strategy, agg_method, temp, tile_k, thr, n_rows, csv_path)
                        )

    # Print summary
    print(f"\n{'='*90}")
    print("MULTI-VIEW SWEEP SUMMARY")
    print(f"{'='*90}")
    fmt = "{:>14} {:>18} {:>5} {:>6} {:>6} {:>8}  {}"
    print(fmt.format("Strategy", "AggMethod", "Temp", "TileK", "Thr", "Images", "CSV"))
    print(
        fmt.format("-" * 14, "-" * 18, "-" * 5, "-" * 6, "-" * 6, "-" * 8, "-" * 40)
    )
    for strategy, agg_method, temp, tile_k, thr, n, path in results_summary:
        print(
            fmt.format(
                strategy, agg_method, f"{temp:.1f}", str(tile_k), f"{thr:.2f}", str(n), path
            )
        )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
