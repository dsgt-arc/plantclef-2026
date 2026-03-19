"""Step 7: Incremental view ablation sweep.

Determines whether each augmentation view independently contributes to F1.
Tests 1-view (no TTA), 2-view (orig + one augmentation), and 3-view (full TTA).

View configs:
  1v:        orig only (no augmentation)
  2v_hflip:  orig + hflip
  2v_rot90:  orig + rot90
  3v:        orig + hflip + rot90 (existing per_view_or, included for validation)

All configs: max_prob + T=1.5 + k={3,4,5,6}
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
    NUM_CLASSES,
    compute_logits_numpy,
    decode_embeddings,
    load_classifier_head,
    _filter_tiles,
)
from utils.aggregator import create_aggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix


# View definitions: (column_name, source_type)
VIEW_SPECS = {
    "orig":  ("teacher_logits_f16", "logits"),
    "hflip": ("embedding_flip_f16", "embedding"),
    "rot90": ("embedding_rot90_f16", "embedding"),
}

# Ablation configs: (label, view_list)
VIEW_CONFIGS = [
    ("1v",        ["orig"]),
    ("2v_hflip",  ["orig", "hflip"]),
    ("2v_rot90",  ["orig", "rot90"]),
    ("3v",        ["orig", "hflip", "rot90"]),
]


def get_view_logits(
    group,
    view_name: str,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
) -> np.ndarray:
    col_name, source_type = VIEW_SPECS[view_name]
    if source_type == "logits":
        return f16_bytes_list_to_f32_matrix(
            group[col_name].tolist(), NUM_CLASSES
        )
    else:
        embeddings = decode_embeddings(group[col_name].tolist())
        return compute_logits_numpy(embeddings, head_weight, head_bias)


def nview_aggregate(
    tile_data,
    views: list,
    agg_method: str,
    config: AggregationConfig,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    image_col: str = "image_name",
):
    """Run per_view_or aggregation with N views. Returns {image_name: probs_f16}."""
    image_topk = config.image_topk if config.image_topk > 0 else None

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

        view_probs = []
        for view_name in views:
            logits = get_view_logits(group, view_name, head_weight, head_bias)
            p_view = aggregator.aggregate(logits).astype(np.float32)
            view_probs.append(p_view)

        p_avg = np.mean(view_probs, axis=0)

        if image_topk is not None:
            k = min(image_topk, p_avg.shape[0])
            if k < p_avg.shape[0]:
                idx = np.argpartition(p_avg, -k)[-k:]
                mask = np.zeros(p_avg.shape[0], dtype=np.bool_)
                mask[idx] = True
                p_avg[~mask] = 0.0

        if geo_mask is not None:
            p_avg[~geo_mask] = 0.0

        results[str(name)] = p_avg.astype(np.float16)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Incremental view ablation: 1v vs 2v vs 3v"
    )

    ap.add_argument("--tile-parquet-root", required=True)
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
        help="Aggregation method (default: max_prob)",
    )
    ap.add_argument("--temperatures", default="1.5")
    ap.add_argument("--tile-filter-ks", default="3,4,5,6")
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

    # Load tile data
    print(f"Loading tile data from {args.tile_parquet_root}...")
    tile_data = load_per_tile_parquet(args.tile_parquet_root)
    print(f"Loaded {len(tile_data)} tile rows")

    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    n_combos = len(VIEW_CONFIGS) * len(temps) * len(tile_ks) * len(thresholds)
    print(
        f"\nSweep: {len(VIEW_CONFIGS)} view-configs x {len(temps)} temps "
        f"x {len(tile_ks)} tile-ks x {len(thresholds)} thresholds "
        f"= {n_combos} submissions"
    )

    results_summary = []

    for view_label, views in VIEW_CONFIGS:
        for temp in temps:
            for tile_k in tile_ks:
                print(f"\n{'='*70}")
                print(
                    f"{view_label}  {args.agg_method}  T={temp}  k={tile_k}  "
                    f"views={views}"
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
                image_probs = nview_aggregate(
                    tile_data,
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
                    f"mv_{view_label}_{args.agg_method}"
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
    print("VIEW ABLATION SWEEP SUMMARY")
    print(f"{'='*90}")
    fmt = "{:>10} {:>6} {:>5} {:>6} {:>6} {:>8}  {}"
    print(fmt.format("Views", "NView", "Temp", "TileK", "Thr", "Images", "CSV"))
    print(
        fmt.format(
            "-" * 10, "-" * 6, "-" * 5, "-" * 6, "-" * 6, "-" * 8, "-" * 50
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
    print(f"\nExisting 3v baseline: per_view_or + max_prob T1.5 k3 = F1 0.31487")


if __name__ == "__main__":
    main()
