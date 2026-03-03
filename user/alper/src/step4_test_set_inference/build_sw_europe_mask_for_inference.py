#!/usr/bin/env python3

import os
import json
import argparse
from typing import Optional, List, Tuple, Set

import numpy as np
import pandas as pd


def load_class_to_species_map(path: str, num_classes: int) -> List[int]:
    """
    Loads a mapping from class_index to species_id.

    Supported formats
      json list: [1395806, 1351284, ...] length == num_classes
      csv with columns: class_index,species_id
      txt one species_id per line
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            raise ValueError("JSON mapping must be a list")
        if len(arr) != num_classes:
            raise ValueError(f"JSON mapping length {len(arr)} does not match num_classes {num_classes}")
        return [int(x) for x in arr]

    if ext == ".csv":
        df = pd.read_csv(path, sep=";")
        if "class_index" not in df.columns or "species_id" not in df.columns:
            raise ValueError("CSV mapping must have columns class_index and species_id")
        df = df.sort_values("class_index")
        if len(df) != num_classes:
            raise ValueError(f"CSV mapping rows {len(df)} does not match num_classes {num_classes}")
        return [int(x) for x in df["species_id"].tolist()]

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) != num_classes:
        raise ValueError(f"TXT mapping lines {len(lines)} does not match num_classes {num_classes}")
    return [int(x) for x in lines]


def load_metadata(path: str, cols: Tuple[str, str, str]) -> pd.DataFrame:
    """
    Reads metadata from CSV or Parquet and returns only the needed columns:
    species_id, latitude, longitude.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";")

    species_col, lat_col, lon_col = cols
    for c in [species_col, lat_col, lon_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c} in {path}")

    out = df[[species_col, lat_col, lon_col]].copy()
    out = out.rename(columns={species_col: "species_id", lat_col: "lat", lon_col: "lon"})
    out["species_id"] = pd.to_numeric(out["species_id"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["species_id", "lat", "lon"])
    out["species_id"] = out["species_id"].astype(np.int64)
    return out


def build_species_set_bbox(df: pd.DataFrame, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Set[int]:
    in_box = (df["lat"] >= lat_min) & (df["lat"] <= lat_max) & (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
    return set(df.loc[in_box, "species_id"].unique().tolist())


def build_species_set_refpoint_nearest_in_bbox(
    df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Set[int]:
    """
    Implements the method you quoted, in a principled way:

    For each species, take its single closest observation to the reference point.
    Keep the species if that closest observation lies inside the bbox.
    """
    dlat = df["lat"].values - ref_lat
    dlon = df["lon"].values - ref_lon
    dist2 = dlat * dlat + dlon * dlon
    df2 = df.copy()
    df2["dist2"] = dist2

    idx = df2.groupby("species_id")["dist2"].idxmin()
    nearest = df2.loc[idx]

    in_box = (
        (nearest["lat"] >= lat_min) & (nearest["lat"] <= lat_max) &
        (nearest["lon"] >= lon_min) & (nearest["lon"] <= lon_max)
    )

    return set(nearest.loc[in_box, "species_id"].unique().tolist())


def build_class_mask(class_to_species: List[int], allowed_species: Set[int]) -> np.ndarray:
    mask = np.zeros((len(class_to_species),), dtype=np.bool_)
    for i, sid in enumerate(class_to_species):
        if sid in allowed_species:
            mask[i] = True
    return mask


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--metadata", required=True, help="Training metadata file CSV or Parquet containing species and lat lon")
    ap.add_argument("--class-to-species", required=True, help="Mapping file from class_index to species_id")
    ap.add_argument("--num-classes", type=int, default=7806)

    ap.add_argument("--species-col", default="species_id")
    ap.add_argument("--lat-col", default="latitude")
    ap.add_argument("--lon-col", default="longitude")

    ap.add_argument("--mode", choices=["bbox_any", "refpoint_nearest"], default="bbox_any")

    ap.add_argument("--lat-min", type=float, default=41.0)
    ap.add_argument("--lat-max", type=float, default=51.5)
    ap.add_argument("--lon-min", type=float, default=-6.5)
    ap.add_argument("--lon-max", type=float, default=11.5)

    ap.add_argument("--ref-lat", type=float, default=44.0)
    ap.add_argument("--ref-lon", type=float, default=4.0)

    ap.add_argument("--out-mask-npy", required=True, help="Output .npy file for bool mask of shape [num_classes]")
    ap.add_argument("--out-species-txt", default=None, help="Optional output text file, one species_id per line")
    ap.add_argument("--out-stats-json", default=None, help="Optional output json file with summary stats")

    args = ap.parse_args()

    class_to_species = load_class_to_species_map(args.class_to_species, args.num_classes)
    df = load_metadata(args.metadata, (args.species_col, args.lat_col, args.lon_col))

    if args.mode == "bbox_any":
        allowed_species = build_species_set_bbox(
            df, args.lat_min, args.lat_max, args.lon_min, args.lon_max
        )
    else:
        allowed_species = build_species_set_refpoint_nearest_in_bbox(
            df,
            args.ref_lat,
            args.ref_lon,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
        )

    mask = build_class_mask(class_to_species, allowed_species)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_mask_npy)), exist_ok=True)
    np.save(args.out_mask_npy, mask)

    if args.out_species_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_species_txt)), exist_ok=True)
        with open(args.out_species_txt, "w") as f:
            for sid in sorted(list(allowed_species)):
                f.write(f"{sid}\n")

    if args.out_stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_stats_json)), exist_ok=True)
        stats = {
            "mode": args.mode,
            "bbox": {
                "lat_min": args.lat_min,
                "lat_max": args.lat_max,
                "lon_min": args.lon_min,
                "lon_max": args.lon_max,
            },
            "ref_point": {"lat": args.ref_lat, "lon": args.ref_lon},
            "num_metadata_rows_used": int(len(df)),
            "num_allowed_species_ids": int(len(allowed_species)),
            "num_allowed_classes": int(mask.sum()),
            "num_classes": int(args.num_classes),
        }
        with open(args.out_stats_json, "w") as f:
            json.dump(stats, f, indent=2)

    print("Mask saved to:", args.out_mask_npy)
    print("Allowed classes:", int(mask.sum()), "out of", args.num_classes)
    print("Allowed species ids:", len(allowed_species))


if __name__ == "__main__":
    main()


    '''
    Mode 1: simplest and usually best starting point

Keep any species that has at least one training observation in a Southwestern Europe bounding box.

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step4_test_set_inference/build_sw_europe_mask_for_inference.py \
  --metadata /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/metadata_PlantCLEF_training_set/PlantCLEF2024singleplanttrainingdata.csv \
  --class-to-species /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --num-classes 7806 \
  --species-col species_id \
  --lat-col latitude \
  --lon-col longitude \
  --mode bbox_any \
  --out-mask-npy /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/species_masking/sw_europe_mask_bool.npy \
  --out-stats-json /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/species_masking/sw_europe_mask_stats.json \
  --out-species-txt /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/species_masking/sw_europe_allowed_species_ids.txt

    
    '''