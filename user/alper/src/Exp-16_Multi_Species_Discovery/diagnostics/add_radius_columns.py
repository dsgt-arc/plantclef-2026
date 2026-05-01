"""Add per-species n_near_<source>_refined_R<radius> columns at
multiple proximity radii (150, 500 km), so we can sweep the radius
parameter at apply time."""
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
REFINED_REF = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
                r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
                r"/outputs/habitat_classifier/refined_source_ref.json")
RADII_KM = (150.0, 200.0, 300.0, 500.0)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = np.radians(lat1)
    b = np.radians(lat2)
    dlat = b - a
    dlon = np.radians(lon2 - lon1)
    h = np.sin(dlat / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def main() -> None:
    with open(REFINED_REF, "r", encoding="utf-8") as f:
        refined = json.load(f)
    print(f"Loaded {len(refined)} refined source centroids")

    df = pd.read_csv(TRAIN_META, sep=";",
                     usecols=["species_id", "latitude", "longitude"],
                     dtype={"species_id": str})
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    valid = df.dropna(subset=["latitude", "longitude"])
    print(f"  {len(valid):,} rows valid lat/lon")

    print("Computing per-species, per-source, per-radius proximity counts...")
    by_sp = valid.groupby("species_id", sort=False)
    rows = []
    for sid, g in by_sp:
        rec = {"species_id": str(sid)}
        for src, c in refined.items():
            d = haversine_km(g["latitude"].values, g["longitude"].values,
                              c["lat"], c["lon"])
            for R_km in RADII_KM:
                col = f"n_near_{src}_refined_R{int(R_km)}"
                rec[col] = int((d <= R_km).sum())
        rows.append(rec)
    add_df = pd.DataFrame(rows)

    profile = pd.read_parquet(GEO_PROFILE)
    drop_cols = [col for col in profile.columns if "_refined_R" in col]
    if drop_cols:
        profile = profile.drop(columns=drop_cols)
    merged = profile.merge(add_df, on="species_id", how="left")
    for src in refined:
        for R_km in RADII_KM:
            col = f"n_near_{src}_refined_R{int(R_km)}"
            merged[col] = merged[col].fillna(0).astype(int)

    merged.to_parquet(GEO_PROFILE, index=False)
    print(f"  wrote -> {GEO_PROFILE}")
    print(f"  added columns: {[c for c in merged.columns if '_refined_R' in c]}")


if __name__ == "__main__":
    main()
