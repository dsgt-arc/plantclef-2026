"""Augment species_geo_profile.parquet with n_near_habitat_<K> columns
for each habitat cluster centroid.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

TRAIN_META = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local"
              r"/PlanCLEF_training_dataset_metadata"
              r"/PlantCLEF2024singleplanttrainingdata.csv")
GEO_PROFILE = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
               r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
               r"/outputs/habitat_classifier/species_geo_profile.parquet")
HAB_CENTROIDS = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
                 r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
                 r"/outputs/habitat_classifier/habitat_centroids.json")
PROXIMITY_RADIUS_KM = 250.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = np.radians(lat1)
    b = np.radians(lat2)
    dlat = b - a
    dlon = np.radians(lon2 - lon1)
    h = np.sin(dlat / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    with open(HAB_CENTROIDS, "r", encoding="utf-8") as f:
        centroids = json.load(f)
    print(f"Loaded {len(centroids)} habitat centroids")

    print("Loading training metadata...")
    df = pd.read_csv(TRAIN_META, sep=";",
                     usecols=["species_id", "latitude", "longitude"],
                     dtype={"species_id": str})
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    valid = df.dropna(subset=["latitude", "longitude"])
    print(f"  {len(valid):,} rows with valid lat/lon")

    print("Computing per-species, per-habitat proximity counts...")
    by_sp = valid.groupby("species_id", sort=False)
    rows = []
    for sid, g in by_sp:
        rec = {"species_id": str(sid)}
        for k, c in centroids.items():
            d = haversine_km(g["latitude"].values, g["longitude"].values,
                              c["lat"], c["lon"])
            rec[f"n_near_habitat_{k}"] = int((d <= PROXIMITY_RADIUS_KM).sum())
        rows.append(rec)
    add_df = pd.DataFrame(rows)

    profile = pd.read_parquet(GEO_PROFILE)
    drop_cols = [c for c in profile.columns if c.startswith("n_near_habitat_")]
    if drop_cols:
        profile = profile.drop(columns=drop_cols)
    merged = profile.merge(add_df, on="species_id", how="left")
    for k in centroids:
        col = f"n_near_habitat_{k}"
        merged[col] = merged[col].fillna(0).astype(int)

    out_path = GEO_PROFILE
    merged.to_parquet(out_path, index=False)
    print(f"  wrote -> {out_path}")
    print(f"  added columns: {[c for c in merged.columns if c.startswith('n_near_habitat_')]}")


if __name__ == "__main__":
    main()
