"""Exp-16 Phase 3 / Gate C — build a filtered training manifest.

Produces a per-image manifest CSV with three filtering knobs:

  1. species filter — keep only images whose species_id passes the
     production-time admissibility mask (geo & aquatic). The per-source
     biogeo mask is NOT applied here, since training is source-agnostic
     and those 5 species ARE admissible on RNNB / coastal sources.

  2. organ filter — keep only `habit, leaf, flower` (default), or any
     user-specified subset. Drops `bark, branch, fruit, scan` which are
     close-up / side-on / flat-scan views misaligned with the
     top-down test-image distribution.

  3. organ-weighted sampling — assign per-image sampling weights
     (e.g. habit=2.0, others=1.0) to bias gradient direction toward
     the test-distribution viewpoint without throwing data away.

Output columns: image_path, species_id, organ, sampling_weight,
                image_backup_url (for download fallback).

Mechanism in one sentence: Restrict Gate C training to admissible
species and to organ views that are biogeographically and visually
consistent with top-down quadrat photos, with optional habit-view
upweighting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METADATA = (
    r"C:\Users\alper\Documents\PlantCLEF_2026_Local"
    r"\PlanCLEF_training_dataset_metadata"
    r"\PlantCLEF2024singleplanttrainingdata.csv"
)
DEFAULT_GEO_MASK = (
    r"C:\Users\alper\Documents\PlantCLEF_2026_Local"
    r"\_GitHub\plantclef-2026\user\alper\src"
    r"\Exp-14_Adaptive_Temporal_Fusion\pace_downloads"
    r"\sw_europe_mask_bool.npy"
)
DEFAULT_AQUATIC_MASK = (
    r"C:\Users\alper\Documents\PlantCLEF_2026_Local"
    r"\_GitHub\plantclef-2026\user\alper\src"
    r"\Exp-14_Adaptive_Temporal_Fusion\outputs\signals"
    r"\aquatic_blacklist_mask.npy"
)
DEFAULT_SPECIES_MAP = (
    r"C:\Users\alper\Documents\PlantCLEF_2026_Local"
    r"\_GitHub\plantclef-2026\user\alper\src"
    r"\Exp-14_Adaptive_Temporal_Fusion\pace_downloads"
    r"\class_to_species_id.json"
)

# Default organ keep-list. `habit` = whole-plant-in-habitat view
# (closest match to top-down quadrat photos). `leaf` = leaf views,
# most are top-down or near-top. `flower` = mixed; many small alpine
# flowers are photographed from above.
DEFAULT_KEEP_ORGANS = ("habit", "leaf", "flower")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default=DEFAULT_METADATA)
    ap.add_argument("--geo-mask", default=DEFAULT_GEO_MASK)
    ap.add_argument("--aquatic-mask", default=DEFAULT_AQUATIC_MASK)
    ap.add_argument("--species-map", default=DEFAULT_SPECIES_MAP,
                    help="JSON list mapping class_idx -> species_id.")
    ap.add_argument("--keep-organs", default=",".join(DEFAULT_KEEP_ORGANS),
                    help="Comma-separated organ values to keep.")
    ap.add_argument("--habit-weight", type=float, default=2.0,
                    help="Per-image sampling weight for organ=habit "
                         "(others get 1.0).")
    ap.add_argument("--learn-tag", default="train",
                    help="Filter by learn_tag column (default 'train').")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-stats-json", default=None)
    args = ap.parse_args()

    print("=" * 72)
    print("Exp-16 Gate C: build_training_manifest")
    print("=" * 72)
    print(f"  metadata:        {args.metadata_csv}")
    print(f"  geo mask:        {args.geo_mask}")
    print(f"  aquatic mask:    {args.aquatic_mask}")
    print(f"  species map:     {args.species_map}")
    print(f"  keep organs:     {args.keep_organs}")
    print(f"  habit weight:    {args.habit_weight}")
    print(f"  learn_tag:       {args.learn_tag}")

    geo = np.load(args.geo_mask).astype(np.bool_)
    aquatic = np.load(args.aquatic_mask).astype(np.bool_)
    if geo.shape != aquatic.shape:
        raise ValueError(f"mask shape mismatch: {geo.shape} vs {aquatic.shape}")
    combined_mask = geo & aquatic
    print(f"\n  geo passes:      {geo.sum()} / {len(geo)}")
    print(f"  aquatic passes:  {aquatic.sum()} / {len(aquatic)}")
    print(f"  combined passes: {combined_mask.sum()} / {len(combined_mask)}")

    with open(args.species_map, encoding="utf-8") as f:
        species_map = json.load(f)
    if not isinstance(species_map, list):
        raise ValueError(f"species_map should be list, got {type(species_map)}")
    admissible_sids = {
        int(species_map[i]) for i in range(len(species_map)) if combined_mask[i]
    }
    print(f"  admissible species_ids: {len(admissible_sids)}")

    print(f"\n  reading metadata...")
    df = pd.read_csv(
        args.metadata_csv,
        sep=";",
        usecols=["image_name", "organ", "species_id", "learn_tag",
                 "image_backup_url"],
    )
    print(f"    total rows: {len(df):,}")

    df = df[df["learn_tag"] == args.learn_tag]
    print(f"    after learn_tag={args.learn_tag}: {len(df):,}")

    df = df[df["species_id"].isin(admissible_sids)]
    print(f"    after species filter: {len(df):,}")

    keep_organs = tuple(o.strip() for o in args.keep_organs.split(",") if o.strip())
    df = df[df["organ"].isin(keep_organs)]
    print(f"    after organ filter ({keep_organs}): {len(df):,}")

    df = df.copy()
    df["sampling_weight"] = np.where(
        df["organ"] == "habit", float(args.habit_weight), 1.0
    )
    df["image_path"] = (
        df["species_id"].astype(str) + "/" + df["image_name"].astype(str)
    )

    out_df = df[["image_path", "species_id", "organ", "sampling_weight",
                 "image_backup_url"]].sort_values("image_path")
    Path(os.path.dirname(os.path.abspath(args.out_manifest))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_manifest, index=False)
    print(f"\n  wrote {len(out_df):,} rows -> {args.out_manifest}")

    organ_counts = df["organ"].value_counts().to_dict()
    print("\n  organ counts (post-filter):")
    for k, v in sorted(organ_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:12s} {v:>10,}")

    n_species_present = df["species_id"].nunique()
    images_per_species = df.groupby("species_id").size()
    print("\n  per-species image counts:")
    print(f"    n species present:  {n_species_present}")
    print(f"    n species missing:  {len(admissible_sids) - n_species_present}")
    print(f"    median images/sp:   {int(images_per_species.median())}")
    print(f"    min images/sp:      {images_per_species.min()}")
    print(f"    max images/sp:      {images_per_species.max()}")
    print(f"    p10 / p90:          "
          f"{int(images_per_species.quantile(0.10))} / "
          f"{int(images_per_species.quantile(0.90))}")
    print(f"    sp with <  10:      {(images_per_species < 10).sum()}")
    print(f"    sp with <  50:      {(images_per_species < 50).sum()}")
    print(f"    sp with < 100:      {(images_per_species < 100).sum()}")

    total_weight = float(df["sampling_weight"].sum())
    avg_weight = total_weight / len(df)
    print(f"\n  total sampling weight: {total_weight:,.0f}")
    print(f"  avg sampling weight:   {avg_weight:.3f}")

    if args.out_stats_json:
        stats = {
            "n_images_post_filter": int(len(df)),
            "n_species_present": int(n_species_present),
            "n_species_admissible": int(len(admissible_sids)),
            "n_species_missing_from_train": int(len(admissible_sids) - n_species_present),
            "organ_counts": {k: int(v) for k, v in organ_counts.items()},
            "median_images_per_species": int(images_per_species.median()),
            "min_images_per_species": int(images_per_species.min()),
            "max_images_per_species": int(images_per_species.max()),
            "p10_images_per_species": int(images_per_species.quantile(0.10)),
            "p90_images_per_species": int(images_per_species.quantile(0.90)),
            "n_species_lt_10_images": int((images_per_species < 10).sum()),
            "n_species_lt_50_images": int((images_per_species < 50).sum()),
            "n_species_lt_100_images": int((images_per_species < 100).sum()),
            "total_sampling_weight": total_weight,
            "habit_weight": float(args.habit_weight),
            "keep_organs": list(keep_organs),
        }
        Path(os.path.dirname(os.path.abspath(args.out_stats_json))).mkdir(
            parents=True, exist_ok=True)
        with open(args.out_stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  wrote stats -> {args.out_stats_json}")


if __name__ == "__main__":
    main()
