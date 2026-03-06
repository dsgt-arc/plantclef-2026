"""Run 4 systematic teacher baseline configs for the paper's baseline table.

T1: No aug filter, all 16 tiles, T=2.0, geo mask
T2: Aug filter, all 16 tiles, best_T, geo mask
T3: Aug filter, top-6 tiles, best_T, geo mask
T4: Aug filter, top-6 tiles, best_T, no geo mask
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass
class BaselineConfig:
    name: str
    aug_filter: bool
    tile_filter_k: int
    temperature: float
    use_geo_mask: bool


def get_baseline_configs(best_temperature: float) -> List[BaselineConfig]:
    return [
        BaselineConfig(
            name="T1",
            aug_filter=False,
            tile_filter_k=16,
            temperature=2.0,
            use_geo_mask=True,
        ),
        BaselineConfig(
            name="T2",
            aug_filter=True,
            tile_filter_k=16,
            temperature=best_temperature,
            use_geo_mask=True,
        ),
        BaselineConfig(
            name="T3",
            aug_filter=True,
            tile_filter_k=6,
            temperature=best_temperature,
            use_geo_mask=True,
        ),
        BaselineConfig(
            name="T4",
            aug_filter=True,
            tile_filter_k=6,
            temperature=best_temperature,
            use_geo_mask=False,
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher baseline table (4 configs)")

    ap.add_argument("--tile-parquet-root", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None, help="Geo species mask .npy")

    ap.add_argument("--best-temperature", type=float, default=2.0)
    ap.add_argument("--topk", type=int, default=10)

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--id-col", default="image_name")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tile data once
    print(f"Loading per-tile data from {args.tile_parquet_root}...")
    tile_data = load_per_tile_parquet(args.tile_parquet_root)
    print(f"Loaded {len(tile_data)} tile rows")

    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    configs = get_baseline_configs(args.best_temperature)
    results: List[Tuple[str, str, int, str]] = []

    for bc in configs:
        print(f"\n{'='*60}")
        print(f"Config {bc.name}: aug_filter={bc.aug_filter}, tile_k={bc.tile_filter_k}, "
              f"T={bc.temperature}, geo_mask={bc.use_geo_mask}")
        print(f"{'='*60}")

        geo_path = args.class_mask_npy if bc.use_geo_mask else None

        config = AggregationConfig(
            temperature=bc.temperature,
            aug_filter=bc.aug_filter,
            tile_filter_k=bc.tile_filter_k,
            geo_mask_path=geo_path,
            image_topk=args.image_topk,
            num_classes=args.num_classes,
        )

        image_probs = run_aggregation(tile_data, config, image_col=args.id_col)

        parquet_path = os.path.join(args.out_dir, f"baseline_{bc.name}.parquet")
        save_aggregated(image_probs, parquet_path, id_col=args.id_col)

        csv_path = os.path.join(args.out_dir, f"baseline_{bc.name}_top{args.topk}.csv")

        # For T4 (no geo mask), also don't apply class_mask in submission
        submission_mask = class_mask if bc.use_geo_mask else None

        n_rows = generate_submission_csv(
            in_parquet=parquet_path,
            out_csv=csv_path,
            species_map=species_map,
            num_classes=args.num_classes,
            class_mask=submission_mask,
            topk=args.topk,
            threshold=0.0,
            min_k=1,
            id_col=args.id_col,
        )

        desc = (
            f"aug={'Y' if bc.aug_filter else 'N'} "
            f"tiles={bc.tile_filter_k} "
            f"T={bc.temperature:.1f} "
            f"geo={'Y' if bc.use_geo_mask else 'N'}"
        )
        results.append((bc.name, desc, n_rows, csv_path))

    # Print summary table
    print(f"\n{'='*80}")
    print("TEACHER BASELINE TABLE")
    print(f"{'='*80}")
    print(f"{'Config':>8} {'Description':>30} {'Images':>8} {'F1 (Kaggle)':>12} CSV")
    print(f"{'-'*8:>8} {'-'*30:>30} {'-'*8:>8} {'-'*12:>12} {'-'*30}")
    for name, desc, n, path in results:
        print(f"{name:>8} {desc:>30} {n:>8d} {'(submit)':>12} {path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
