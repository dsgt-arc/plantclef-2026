"""Re-aggregate per-tile data with multiple temperatures, generate a submission for each."""
from __future__ import annotations

import argparse
import os

import numpy as np

from phase2.aggregate_tiles import (
    AggregationConfig,
    load_per_tile_parquet,
    run_aggregation,
    save_aggregated,
)
from phase2.generate_submission import (
    generate_submission_csv,
    load_class_mask_npy,
    load_species_map,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Temperature sweep for teacher aggregation")

    ap.add_argument("--tile-parquet-root", required=True, help="Phase 1 per-tile parquet root")
    ap.add_argument("--out-dir", required=True, help="Output directory for sweep results")

    ap.add_argument("--species-map", required=True, help="Class-to-species JSON/CSV")
    ap.add_argument("--class-mask-npy", default=None, help="Geo species mask .npy")

    ap.add_argument(
        "--temperatures",
        default="1.0,1.5,2.0,3.0",
        help="Comma-separated temperature values (default: 1.0,1.5,2.0,3.0)",
    )
    ap.add_argument("--aug-filter", action="store_true", default=True)
    ap.add_argument("--no-aug-filter", action="store_false", dest="aug_filter")
    ap.add_argument("--tile-filter-k", type=int, default=16, help="Keep top-K tiles (16=all)")
    ap.add_argument("--topk", type=int, default=10, help="Top-K species per image in submission")

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--id-col", default="image_name")

    args = ap.parse_args()

    temps = [float(t.strip()) for t in args.temperatures.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tile data once
    print(f"Loading per-tile data from {args.tile_parquet_root}...")
    tile_data = load_per_tile_parquet(args.tile_parquet_root)
    print(f"Loaded {len(tile_data)} tile rows")

    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    results_summary = []

    for temp in temps:
        print(f"\n{'='*60}")
        print(f"Temperature = {temp}")
        print(f"{'='*60}")

        config = AggregationConfig(
            temperature=temp,
            aug_filter=args.aug_filter,
            tile_filter_k=args.tile_filter_k,
            geo_mask_path=args.class_mask_npy,
            image_topk=args.image_topk,
            num_classes=args.num_classes,
        )

        image_probs = run_aggregation(tile_data, config, image_col=args.id_col)

        # Save aggregated parquet
        parquet_path = os.path.join(args.out_dir, f"sweep_T{temp:.1f}.parquet")
        save_aggregated(image_probs, parquet_path, id_col=args.id_col)

        # Generate submission CSV
        csv_path = os.path.join(args.out_dir, f"sweep_T{temp:.1f}_top{args.topk}.csv")
        n_rows = generate_submission_csv(
            in_parquet=parquet_path,
            out_csv=csv_path,
            species_map=species_map,
            num_classes=args.num_classes,
            class_mask=class_mask,
            topk=args.topk,
            threshold=0.0,
            min_k=1,
            id_col=args.id_col,
        )

        results_summary.append((temp, n_rows, csv_path))

    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Temperature':>12} {'Images':>8} {'CSV Path'}")
    print(f"{'-'*12:>12} {'-'*8:>8} {'-'*40}")
    for temp, n, path in results_summary:
        print(f"{temp:>12.1f} {n:>8d} {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
