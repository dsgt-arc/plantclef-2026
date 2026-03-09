#!/usr/bin/env python3
"""Build a geographic species mask for PlantCLEF inference.

Two-step approach (following the geopy reference notebook):
  1. For each species, find its single closest observation to a reference
     point using squared Euclidean distance on (lat, lon).
  2. Keep only species whose closest point falls inside the union of
     target country polygons (Natural Earth shapefile).

Outputs:
  - Boolean numpy mask  [num_classes]  (.npy)
  - Allowed species IDs (.txt, one per line)
  - Summary statistics   (.json)
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)


# ── defaults ────────────────────────────────────────────────────────────
DEFAULT_REF_LAT = 44.0
DEFAULT_REF_LON = 4.0
DEFAULT_COUNTRIES = ["France", "Spain", "Italy", "Switzerland"]
DEFAULT_NUM_CLASSES = 7806


# ── shapefile download ──────────────────────────────────────────────────

def download_natural_earth(dest_dir: str) -> str:
    """Download and extract the Natural Earth 110m countries shapefile.

    Returns the path to the .shp file.
    """
    dest = Path(dest_dir)
    shp = dest / "ne_110m_admin_0_countries.shp"
    if shp.exists():
        print(f"Shapefile already exists: {shp}")
        return str(shp)

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ne_110m_admin_0_countries.zip"

    print(f"Downloading Natural Earth shapefile to {zip_path}...")
    urllib.request.urlretrieve(NATURAL_EARTH_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()

    if not shp.exists():
        raise FileNotFoundError(f"Expected {shp} after extraction")
    print(f"Shapefile ready: {shp}")
    return str(shp)


# ── helpers ─────────────────────────────────────────────────────────────

def load_class_to_species_map(path: str, num_classes: int) -> list[int]:
    """Load class-index → species-id mapping (JSON list, CSV, or TXT)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path) as f:
            arr = json.load(f)
        if len(arr) != num_classes:
            raise ValueError(
                f"JSON mapping length {len(arr)} != num_classes {num_classes}"
            )
        return [int(x) for x in arr]

    if ext == ".csv":
        df = pd.read_csv(path, sep=";")
        df = df.sort_values("class_index")
        if len(df) != num_classes:
            raise ValueError(
                f"CSV rows {len(df)} != num_classes {num_classes}"
            )
        return [int(x) for x in df["species_id"].tolist()]

    # plain text, one species_id per line
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) != num_classes:
        raise ValueError(
            f"TXT lines {len(lines)} != num_classes {num_classes}"
        )
    return [int(x) for x in lines]


def load_metadata(
    path: str,
    species_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """Read training metadata and return (species_id, lat, lon) only."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        # try comma first, fall back to semicolon
        df = pd.read_csv(path, sep=";")
        if species_col not in df.columns:
            df = pd.read_csv(path, sep=",")

    for c in (species_col, lat_col, lon_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")

    out = df[[species_col, lat_col, lon_col]].copy()
    out.columns = ["species_id", "lat", "lon"]
    out["species_id"] = pd.to_numeric(out["species_id"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["species_id", "lat", "lon"])
    out["species_id"] = out["species_id"].astype(np.int64)
    return out


# ── step 1: closest observation per species ─────────────────────────────

def closest_point_per_species(
    df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
) -> pd.DataFrame:
    """For each species, keep only the observation closest to (ref_lat, ref_lon).

    Uses squared Euclidean distance on raw (lat, lon) — same metric as the
    geopy reference notebook.
    """
    df = df.copy()
    df["dist2"] = (df["lat"] - ref_lat) ** 2 + (df["lon"] - ref_lon) ** 2
    idx = df.groupby("species_id")["dist2"].idxmin()
    return df.loc[idx, ["species_id", "lat", "lon"]].reset_index(drop=True)


# ── step 2: point-in-polygon with country boundaries ───────────────────

def load_country_union(shapefile: str, countries: list[str]):
    """Load Natural Earth shapefile and return the union geometry of the
    requested countries."""
    world = gpd.read_file(shapefile)
    selected = world[world["NAME"].isin(countries)]
    if selected.empty:
        available = sorted(world["NAME"].unique().tolist())
        raise ValueError(
            f"None of {countries} found in shapefile. "
            f"Available: {available[:20]}..."
        )
    found = sorted(selected["NAME"].unique().tolist())
    print(f"Country polygons loaded: {found}")
    return selected.unary_union


def filter_species_in_polygon(
    nearest_df: pd.DataFrame,
    polygon_union,
) -> set[int]:
    """Keep species whose closest point falls inside the country union."""
    gdf = gpd.GeoDataFrame(
        nearest_df,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(nearest_df["lon"], nearest_df["lat"])
        ],
        crs="EPSG:4326",
    )
    inside = gdf[gdf.geometry.within(polygon_union)]
    return set(inside["species_id"].astype(int).tolist())


# ── build mask array ────────────────────────────────────────────────────

def build_class_mask(
    class_to_species: list[int],
    allowed_species: set[int],
) -> np.ndarray:
    mask = np.zeros(len(class_to_species), dtype=np.bool_)
    for i, sid in enumerate(class_to_species):
        if sid in allowed_species:
            mask[i] = True
    return mask


# ── CLI ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build geographic species mask (closest-point + country polygon).",
    )

    # inputs
    p.add_argument(
        "--metadata", required=True,
        help="Training metadata CSV/Parquet with species, lat, lon columns",
    )
    p.add_argument(
        "--class-to-species", required=True,
        help="Mapping file (JSON list / CSV / TXT) from class index to species_id",
    )
    p.add_argument(
        "--shapefile", default=None,
        help="Natural Earth admin-0 countries shapefile (.shp). "
             "If omitted, downloads automatically to --shapefile-dir.",
    )
    p.add_argument(
        "--shapefile-dir", default=None,
        help="Directory to download/cache the Natural Earth shapefile. "
             "Defaults to <out-mask-npy parent>/ne_110m_admin_0_countries",
    )

    # column names
    p.add_argument("--species-col", default="species_id")
    p.add_argument("--lat-col", default="latitude")
    p.add_argument("--lon-col", default="longitude")

    # parameters
    p.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    p.add_argument("--ref-lat", type=float, default=DEFAULT_REF_LAT)
    p.add_argument("--ref-lon", type=float, default=DEFAULT_REF_LON)
    p.add_argument(
        "--countries", nargs="+", default=DEFAULT_COUNTRIES,
        help="Country names to include (default: France Spain Italy Switzerland)",
    )

    # outputs
    p.add_argument("--out-mask-npy", required=True, help="Output boolean mask .npy")
    p.add_argument("--out-species-txt", default=None, help="Output allowed species list")
    p.add_argument("--out-stats-json", default=None, help="Output summary stats JSON")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── resolve shapefile ───────────────────────────────────────────
    if args.shapefile and os.path.exists(args.shapefile):
        shapefile_path = args.shapefile
    else:
        dl_dir = args.shapefile_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.out_mask_npy)),
            "ne_110m_admin_0_countries",
        )
        shapefile_path = download_natural_earth(dl_dir)

    # ── load inputs ─────────────────────────────────────────────────
    print("Loading class-to-species mapping...")
    class_to_species = load_class_to_species_map(
        args.class_to_species, args.num_classes,
    )

    print(f"Loading metadata from {args.metadata}...")
    df = load_metadata(
        args.metadata, args.species_col, args.lat_col, args.lon_col,
    )
    n_rows = len(df)
    n_species_total = df["species_id"].nunique()
    print(f"  {n_rows:,} rows, {n_species_total:,} unique species (after dropping null coords)")

    # ── step 1: closest point per species ───────────────────────────
    print(f"Finding closest observation per species to ({args.ref_lat}, {args.ref_lon})...")
    nearest = closest_point_per_species(df, args.ref_lat, args.ref_lon)
    print(f"  {len(nearest):,} species with representative points")

    # ── step 2: point-in-polygon ────────────────────────────────────
    print(f"Loading country polygons: {args.countries}...")
    polygon_union = load_country_union(shapefile_path, args.countries)

    print("Filtering species by country boundaries...")
    allowed_species = filter_species_in_polygon(nearest, polygon_union)
    print(f"  {len(allowed_species):,} species pass geographic filter")

    # ── build mask ──────────────────────────────────────────────────
    mask = build_class_mask(class_to_species, allowed_species)
    print(f"  {mask.sum():,} / {args.num_classes} classes allowed in mask")

    # ── save outputs ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_mask_npy)), exist_ok=True)
    np.save(args.out_mask_npy, mask)
    print(f"Mask saved: {args.out_mask_npy}")

    if args.out_species_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_species_txt)), exist_ok=True)
        with open(args.out_species_txt, "w") as f:
            for sid in sorted(allowed_species):
                f.write(f"{sid}\n")
        print(f"Species list saved: {args.out_species_txt}")

    if args.out_stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_stats_json)), exist_ok=True)
        stats = {
            "method": "closest_point_country_polygon",
            "ref_point": {"lat": args.ref_lat, "lon": args.ref_lon},
            "countries": args.countries,
            "num_metadata_rows": n_rows,
            "num_species_in_metadata": n_species_total,
            "num_allowed_species": len(allowed_species),
            "num_allowed_classes": int(mask.sum()),
            "num_classes": args.num_classes,
        }
        with open(args.out_stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved: {args.out_stats_json}")


if __name__ == "__main__":
    main()



'''

 OUT_DIR="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/aerten3/species_masking"

  python phase1/build_geo_species_mask.py \
    --metadata
  "$PLANTCLEF_ROOT/data/raw/metadata_PlantCLEF_training_set/PlantCLEF2024singleplanttrainingdata.csv" \
    --class-to-species "$PLANTCLEF_ROOT/data/raw/class_id_to_species_mapping/class_to_species_id.json" \
    --countries France Spain Italy Switzerland \
    --out-mask-npy "$OUT_DIR/sw_europe_mask_bool.npy" \
    --out-species-txt "$OUT_DIR/sw_europe_allowed_species_ids.txt" \
    --out-stats-json "$OUT_DIR/sw_europe_mask_stats.json"



'''