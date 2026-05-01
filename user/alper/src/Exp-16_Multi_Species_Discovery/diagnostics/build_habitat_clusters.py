"""Cluster species into habitat groups using their per-test-source
training-data proximity profile.

Approach: each species has frac_near_<source> for the 8 test sources.
This 8-dim vector is a natural fingerprint for which French/SW-European
ecological zones the species belongs to. Cluster these vectors.

Filter out species with no SW-Europe training presence before clustering
(otherwise tropical/global species pollute the clustering).

Output: outputs/habitat_classifier/habitat_clusters.parquet
        outputs/habitat_classifier/habitat_centroids.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

GEO_PROFILE = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
               r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
               r"/outputs/habitat_classifier/species_geo_profile.parquet")
SPECIES_NAME = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
                r"/plantclef-2026/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
                r"/pace_downloads/species_id_to_name.json")
OUT_DIR = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
           r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
           r"/outputs/habitat_classifier")
SOURCES = ("CBN-PdlC", "CBN-Pla", "CBN-can", "GUARDEN", "LISAH",
           "OPTMix", "RNNB", "2024-CEV3")
N_HABITATS = 5

# Source coordinates from earlier file (matching apply_habitat_fit.py)
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


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(GEO_PROFILE)
    print(f"Loaded {len(df)} species profiles")

    frac_cols = [f"frac_near_{s}" for s in SOURCES]
    n_near_cols = [f"n_near_{s}" for s in SOURCES]

    # Filter to species with SW-Europe presence: total n_near across all
    # 8 sources >= 5 (any species seen at least 5 times near any of our
    # test sources). This excludes tropical/global outliers.
    sw_total = df[n_near_cols].sum(axis=1)
    valid = df[sw_total >= 5].copy()
    print(f"  {len(valid)} species with >=5 training obs near any test source")

    # Cluster on frac_near_<source> features (each species' 8-dim
    # proximity profile, normalized).
    X = valid[frac_cols].values.astype(np.float32)
    print(f"  feature matrix: {X.shape}")

    print(f"\nFitting KMeans k={N_HABITATS}...")
    km = KMeans(n_clusters=N_HABITATS, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    valid["habitat"] = labels

    # Compute per-habitat: which test sources is this habitat closest to?
    centroids = {}
    for k in range(N_HABITATS):
        sub = valid[valid["habitat"] == k]
        # Centroid in (lat, lon, alt) space — for the apply step
        c = {
            "lat": float(sub["lat_median"].median()),
            "lon": float(sub["lon_median"].median()),
            "alt": float(sub["alt_median"].median()),
            "n_species": int(len(sub)),
            "alt_lo": float(sub["alt_p25"].median()),
            "alt_hi": float(sub["alt_p75"].median()),
        }
        # Find the dominant test source (the one this habitat is most
        # associated with), and the second-most-dominant (for label).
        avg_frac = {s: float(sub[f"frac_near_{s}"].mean()) for s in SOURCES}
        # Sort by avg_frac
        ranked = sorted(avg_frac.items(), key=lambda kv: -kv[1])
        c["dominant_source"] = ranked[0][0]
        c["dominant_frac"] = ranked[0][1]
        c["second_source"] = ranked[1][0]
        c["second_frac"] = ranked[1][1]
        centroids[k] = c
        print(f"  habitat {k}: dom={c['dominant_source']} "
              f"({c['dominant_frac']:.2f})  2nd={c['second_source']} "
              f"({c['second_frac']:.2f})  lat={c['lat']:.2f}  "
              f"lon={c['lon']:.2f}  alt={c['alt']:.0f}  n_species={c['n_species']}")

    # Sanity: print 5 most-trained species per habitat
    with open(SPECIES_NAME, "r", encoding="utf-8") as f:
        sid_to_name = json.load(f)

    print("\nSanity — 5 most-trained species per habitat cluster:")
    for k in range(N_HABITATS):
        sub = valid[valid["habitat"] == k].nlargest(5, "n_obs")
        print(f"\n  habitat {k} (dom={centroids[k]['dominant_source']}, "
              f"alt={centroids[k]['alt']:.0f}):")
        for _, r in sub.iterrows():
            nm = sid_to_name.get(str(r["species_id"]), "?")
            print(f"    {nm}  n_obs={r['n_obs']}  alt={r['alt_median']:.0f}  "
                  f"lat={r['lat_median']:.1f}  lon={r['lon_median']:.1f}")

    # Save
    out = valid[["species_id", "habitat", "lat_median", "lon_median",
                 "alt_median", "n_obs"]].copy()
    out_path = os.path.join(OUT_DIR, "habitat_clusters.parquet")
    out.to_parquet(out_path, index=False)
    print(f"\n  wrote -> {out_path}")

    cent_path = os.path.join(OUT_DIR, "habitat_centroids.json")
    with open(cent_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in centroids.items()}, f, indent=2)
    print(f"  wrote -> {cent_path}")


if __name__ == "__main__":
    main()
