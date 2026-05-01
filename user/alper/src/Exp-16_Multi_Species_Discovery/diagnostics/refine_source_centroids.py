"""Test 2 — Refine source centroids using training-data anchor species.

Replace hand-set lat/lon centroids with data-derived centroids:
1. For each source, pick the top-N predicted species in the locked
   Stage 13 + N3_r0.5 winner (these are the source's "anchor" flora).
2. Filter to anchors that are NOT also in the top-N of >=3 other
   sources (i.e. drop cosmopolitan species that appear everywhere).
3. Compute the median (lat, lon, alt) of those anchor species'
   TRAINING observations as the new centroid.

Output: refined_source_ref.json + a fresh species_geo_profile_refined.parquet
that has n_near_<source>_refined columns using the new centroids.
"""
from __future__ import annotations

import ast
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

WINNER = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
          r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
          r"/outputs/n_floor_sweep/N3_r0.5_WINNER.csv")
TRAIN_META = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local"
              r"/PlanCLEF_training_dataset_metadata"
              r"/PlantCLEF2024singleplanttrainingdata.csv")
GEO_PROFILE = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
               r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
               r"/outputs/habitat_classifier/species_geo_profile.parquet")
OUT_DIR = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
           r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
           r"/outputs/habitat_classifier")

ALL_SRC_PREFIXES = (
    "RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix",
    "2024-CEV3", "CBN-can",
)
ANCHORS_PER_SOURCE = 30
COSMOPOLITAN_MAX_SOURCES = 2  # appears in <=2 sources' top-30 to count
PROXIMITY_RADIUS_KM = 250.0


def source_of(qid: str) -> str:
    for p in ALL_SRC_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = np.radians(lat1)
    b = np.radians(lat2)
    dlat = b - a
    dlon = np.radians(lon2 - lon1)
    h = np.sin(dlat / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Step 1: collect per-source top species from locked winner
    print("Loading locked winner CSV...")
    df = pd.read_csv(WINNER)
    df["source"] = df["quadrat_id"].apply(source_of)
    df["sids"] = df["species_ids"].apply(ast.literal_eval)

    per_src_counts: Dict[str, Counter] = defaultdict(Counter)
    for _, r in df.iterrows():
        for sid in r["sids"]:
            per_src_counts[r["source"]][str(sid)] += 1

    # Step 2: identify cosmopolitans (in top-N of many sources)
    species_appearance_count: Counter = Counter()
    per_src_top: Dict[str, List[str]] = {}
    for src, counts in per_src_counts.items():
        top = [s for s, _ in counts.most_common(ANCHORS_PER_SOURCE)]
        per_src_top[src] = top
        for sid in top:
            species_appearance_count[sid] += 1

    # Step 3: per-source anchor species = top-N excluding cosmopolitans
    print("\nPer-source anchor species (after dropping cosmopolitans):")
    anchors_by_src: Dict[str, List[str]] = {}
    for src in ALL_SRC_PREFIXES:
        top = per_src_top.get(src, [])
        anchors = [s for s in top
                   if species_appearance_count[s] <= COSMOPOLITAN_MAX_SOURCES]
        anchors_by_src[src] = anchors
        print(f"  {src:<12} top={len(top)}  anchors={len(anchors)}")

    # Step 4: load training metadata
    print("\nLoading training metadata...")
    train = pd.read_csv(TRAIN_META, sep=";",
                        usecols=["species_id", "latitude", "longitude",
                                 "altitude"],
                        dtype={"species_id": str})
    train["latitude"] = pd.to_numeric(train["latitude"], errors="coerce")
    train["longitude"] = pd.to_numeric(train["longitude"], errors="coerce")
    train["altitude"] = pd.to_numeric(train["altitude"], errors="coerce")
    train_ll = train.dropna(subset=["latitude", "longitude"])
    print(f"  {len(train_ll):,} rows valid lat/lon")

    # Restrict to SW Europe (lat 35-52, lon -10 to 10) so anchor centroids
    # don't get pulled by training photos in tropical/global regions.
    sw = train_ll[(train_ll["latitude"].between(35.0, 52.0)) &
                   (train_ll["longitude"].between(-10.0, 10.0))]
    print(f"  {len(sw):,} rows within SW-Europe bounds")

    # Step 5: compute refined centroid per source as median (lat, lon, alt)
    # of anchor-species training observations
    print("\nRefined centroids:")
    refined: Dict[str, dict] = {}
    for src, anchors in anchors_by_src.items():
        sub = sw[sw["species_id"].isin(anchors)]
        if len(sub) == 0:
            print(f"  {src:<12} no anchor-species training obs in SW Europe; "
                  f"keeping hand-set value")
            continue
        c = {
            "lat": float(sub["latitude"].median()),
            "lon": float(sub["longitude"].median()),
            "alt_lo": float(sub["altitude"].quantile(0.10)),
            "alt_hi": float(sub["altitude"].quantile(0.90)),
            "n_anchor_obs": int(len(sub)),
            "n_anchors": len(anchors),
        }
        refined[src] = c
        print(f"  {src:<12} lat={c['lat']:.2f}  lon={c['lon']:.2f}  "
              f"alt=[{c['alt_lo']:.0f},{c['alt_hi']:.0f}]  "
              f"n_anchor_obs={c['n_anchor_obs']}  n_anchors={c['n_anchors']}")

    # Save refined ref centroids
    out_path = os.path.join(OUT_DIR, "refined_source_ref.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)
    print(f"\n  wrote -> {out_path}")

    # Step 6: rebuild per-species n_near_<source>_refined columns
    print("\nRebuilding per-species n_near_<source>_refined counts...")
    by_sp = train_ll.groupby("species_id", sort=False)
    rows = []
    for sid, g in by_sp:
        rec = {"species_id": str(sid)}
        for src, c in refined.items():
            d = haversine_km(g["latitude"].values, g["longitude"].values,
                              c["lat"], c["lon"])
            rec[f"n_near_{src}_refined"] = int((d <= PROXIMITY_RADIUS_KM).sum())
        rows.append(rec)
    add_df = pd.DataFrame(rows)

    profile = pd.read_parquet(GEO_PROFILE)
    drop_cols = [col for col in profile.columns if col.endswith("_refined")]
    if drop_cols:
        profile = profile.drop(columns=drop_cols)
    merged = profile.merge(add_df, on="species_id", how="left")
    for src in refined:
        col = f"n_near_{src}_refined"
        merged[col] = merged[col].fillna(0).astype(int)

    merged.to_parquet(GEO_PROFILE, index=False)
    print(f"  wrote -> {GEO_PROFILE}")
    print(f"  added columns: {[c for c in merged.columns if c.endswith('_refined')]}")


if __name__ == "__main__":
    main()
