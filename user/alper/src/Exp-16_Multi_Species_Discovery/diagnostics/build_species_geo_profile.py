"""Build per-species geographic profile from PlantCLEF2024 training data.

For each species_id, compute:
  - n_obs_total
  - median altitude, p25, p75
  - median latitude, longitude
  - centroid stats per-source (count of training observations within
    a radius of each test source's reference centroid)

Output: parquet with one row per species_id, columns include altitude
quartiles, lat/lon centroid, and per-source proximity counts.

The "external information" this experiment imposes on the test
predictions: the *training-data geographic distribution* of each
species. Cosmopolitan species (high counts everywhere) are not
demoted; range-restricted species (zero counts near a particular
source) get demoted on that source.

Reference centroids per source are set from audit knowledge:
  - CBN-PdlC: Pyrenees (lat 42.7, lon 0.5, alt-range 1500-3300m)
  - CBN-Pla: Alps      (lat 44.5, lon 6.5, alt-range 1500-3300m)
  - CBN-can: Cantal    (lat 45.0, lon 2.7, alt-range 900-2500m)
  - GUARDEN: Med coast (lat 43.5, lon 4.0, alt-range 0-200m)
  - RNNB:    Med coast (lat 43.5, lon 4.0, alt-range 0-200m)
  - LISAH:   Montpellr (lat 43.6, lon 3.9, alt-range 0-400m)
  - OPTMix:  C. France (lat 47.5, lon 2.5, alt-range 50-600m)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_META = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local"
              r"/PlanCLEF_training_dataset_metadata"
              r"/PlantCLEF2024singleplanttrainingdata.csv")
SPECIES_NAME = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
                r"/plantclef-2026/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
                r"/pace_downloads/species_id_to_name.json")
CLASS_TO_SID = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
                r"/plantclef-2026/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
                r"/pace_downloads/class_to_species_id.json")
OUT_DIR = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
           r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
           r"/outputs/habitat_classifier")

# Per-source expected geographic locale
SOURCE_REF = {
    "CBN-PdlC":   {"lat": 42.7, "lon": 0.5,  "alt_lo": 1500, "alt_hi": 3300},
    "CBN-Pla":    {"lat": 44.5, "lon": 6.5,  "alt_lo": 1500, "alt_hi": 3300},
    "CBN-can":    {"lat": 45.0, "lon": 2.7,  "alt_lo": 900,  "alt_hi": 2500},
    "GUARDEN":    {"lat": 43.5, "lon": 4.0,  "alt_lo": 0,    "alt_hi": 200},
    "RNNB":       {"lat": 43.5, "lon": 4.0,  "alt_lo": 0,    "alt_hi": 200},
    "LISAH":      {"lat": 43.6, "lon": 3.9,  "alt_lo": 0,    "alt_hi": 400},
    "OPTMix":     {"lat": 47.5, "lon": 2.5,  "alt_lo": 50,   "alt_hi": 800},
    "2024-CEV3":  {"lat": 45.0, "lon": 6.0,  "alt_lo": 1500, "alt_hi": 3000},
}
PROXIMITY_RADIUS_KM = 250.0  # same biogeographic region


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

    print("Loading training metadata...")
    cols_needed = ["species_id", "altitude", "latitude", "longitude"]
    df = pd.read_csv(TRAIN_META, sep=";", usecols=cols_needed,
                     dtype={"species_id": str})
    print(f"  {len(df):,} rows")

    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    valid = df.dropna(subset=["latitude", "longitude"])
    print(f"  {len(valid):,} rows with valid lat/lon")

    # Per-species summary
    print("\nComputing per-species geographic profile...")
    out_rows = []
    by_sp = valid.groupby("species_id", sort=False)
    for sid, g in by_sp:
        rec = {"species_id": str(sid), "n_obs": len(g)}
        if "altitude" in g and g["altitude"].notna().sum() > 0:
            alt = g["altitude"].dropna()
            rec["alt_median"] = float(alt.median())
            rec["alt_p25"] = float(alt.quantile(0.25))
            rec["alt_p75"] = float(alt.quantile(0.75))
            rec["alt_n"] = int(len(alt))
        else:
            rec["alt_median"] = float("nan")
            rec["alt_p25"] = float("nan")
            rec["alt_p75"] = float("nan")
            rec["alt_n"] = 0
        rec["lat_median"] = float(g["latitude"].median())
        rec["lon_median"] = float(g["longitude"].median())
        # Per-source proximity counts (training obs within RADIUS km of source)
        for src, ref in SOURCE_REF.items():
            d = haversine_km(g["latitude"].values, g["longitude"].values,
                              ref["lat"], ref["lon"])
            n_near = int((d <= PROXIMITY_RADIUS_KM).sum())
            rec[f"n_near_{src}"] = n_near
            rec[f"frac_near_{src}"] = n_near / max(len(g), 1)
        out_rows.append(rec)
    out_df = pd.DataFrame(out_rows)
    print(f"  built profile for {len(out_df)} species")

    out_path = os.path.join(OUT_DIR, "species_geo_profile.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"  wrote -> {out_path}")

    # Quick sanity: print a few species' profiles
    print("\nSanity — top alpine indicator species near CBN-PdlC:")
    sub = out_df[(out_df["alt_median"] > 1500)
                 & (out_df["frac_near_CBN-PdlC"] > 0.3)].nlargest(8, "n_obs")
    with open(SPECIES_NAME, "r", encoding="utf-8") as f:
        sid_to_name = json.load(f)
    for _, r in sub.iterrows():
        nm = sid_to_name.get(str(r["species_id"]), "?")
        print(f"  sid={r['species_id']:>8}  {nm:<35}  "
              f"n={r['n_obs']:>5}  alt_med={r['alt_median']:.0f}  "
              f"frac_near_PdlC={r['frac_near_CBN-PdlC']:.2f}")

    print("\nSanity — top coastal Mediterranean species near GUARDEN:")
    sub = out_df[(out_df["alt_median"] < 100)
                 & (out_df["frac_near_GUARDEN"] > 0.3)
                 & (out_df["lat_median"] < 44)].nlargest(8, "n_obs")
    for _, r in sub.iterrows():
        nm = sid_to_name.get(str(r["species_id"]), "?")
        print(f"  sid={r['species_id']:>8}  {nm:<35}  "
              f"n={r['n_obs']:>5}  alt_med={r['alt_median']:.0f}  "
              f"frac_near_GUARDEN={r['frac_near_GUARDEN']:.2f}")


if __name__ == "__main__":
    main()
