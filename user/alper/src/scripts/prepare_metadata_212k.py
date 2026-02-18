import pandas as pd
import re
import argparse
from pathlib import Path

HASH_PATTERN = re.compile(r"(.+?)_[a-f0-9]{10}\.jpg$", re.IGNORECASE)

USEFUL_COLS = [
    "point_id",
    "year",
    "th_lat",
    "th_long",
    "gps_altitude",
    "lc1_label",
    "file_path_ftp_cover",
]

OUTPUT_COLS = [
    "image_name",
    "point_id",
    "year",
    "latitude",
    "longitude",
    "altitude",
    "land_cover",
]

def build_local_image_map(image_dir: Path) -> dict:
    """
    Returns dict: match_key (original stem lower) -> actual filename in image_dir
    Handles both hashed and non hashed filenames.
    """
    local_image_map = {}
    jpgs = list(image_dir.glob("*.jpg"))

    for p in jpgs:
        name = p.name
        m = HASH_PATTERN.match(name)
        if m:
            key = m.group(1).lower().strip()
        else:
            key = p.stem.lower().strip()
        local_image_map[key] = name

    return local_image_map

def filename_to_key_from_url_series(s: pd.Series) -> pd.Series:
    # Take last path part, strip extension, normalize
    return (
        s.astype(str)
         .str.split("/")
         .str[-1]
         .str.replace(".jpg", "", case=False, regex=False)
         .str.lower()
         .str.strip()
    )

def main():
    parser = argparse.ArgumentParser(description="Metadata matcher for LUCAS images using streaming CSV reads")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing the .jpg images")
    parser.add_argument("--attr_csv", type=str, required=True, help="Path to lucas_cover_attr.csv")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Rows per chunk when reading attr CSV")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    attr_csv = Path(args.attr_csv)
    output_path = Path(args.output)

    print(f"Scanning image folder: {image_dir}")
    local_image_map = build_local_image_map(image_dir)
    image_keys = set(local_image_map.keys())
    print(f"Found {len(local_image_map)} images")

    print(f"Streaming attribute CSV: {attr_csv}")
    print(f"Chunksize: {args.chunksize}")

    kept_chunks = []
    total_seen = 0
    total_matched = 0

    reader = pd.read_csv(
        attr_csv,
        usecols=USEFUL_COLS,
        low_memory=False,
        chunksize=args.chunksize,
    )

    for i, chunk in enumerate(reader, start=1):
        total_seen += len(chunk)

        chunk["match_key"] = filename_to_key_from_url_series(chunk["file_path_ftp_cover"])
        matched = chunk[chunk["match_key"].isin(image_keys)].copy()
        total_matched += len(matched)

        if len(matched) > 0:
            kept_chunks.append(matched)

        if i % 5 == 0:
            print(f"Processed {total_seen:,} rows, matched {total_matched:,} rows so far")

    if not kept_chunks:
        raise RuntimeError("No matches found. Check that image filenames and file_path_ftp_cover naming align.")

    df = pd.concat(kept_chunks, ignore_index=True)

    # Deduplicate by match_key so each image appears once
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["match_key"], keep="first")
    print(f"Matched rows before dedup: {before_dedup:,}, after dedup: {len(df):,}")

    # Add the actual local filename for your image set
    df["image_name"] = df["match_key"].map(local_image_map)

    # Coordinates
    df["latitude"] = df["th_lat"]
    df["longitude"] = df["th_long"]

    # Fill fields
    df["altitude"] = df["gps_altitude"].fillna(0.0)
    df["land_cover"] = df["lc1_label"].fillna("Unknown")

    # Drop missing coordinates
    before_geo = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    dropped_geo = before_geo - len(df)
    print(f"Dropped {dropped_geo:,} rows missing latitude or longitude")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLS].to_csv(output_path, index=False)

    print(f"SUCCESS wrote: {output_path}")
    print(f"Final rows: {len(df):,}")

if __name__ == "__main__":
    main()
