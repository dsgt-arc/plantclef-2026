#!/usr/bin/env python3
"""
Majority vote aggregation + submission CSV from per-tile species.

Reads parquet produced by extract_testdata_tile_species.py and generates
submission CSV matching the reference paper pipeline:

  1. Read per-tile species from parquet (softmax + top-K already applied in PyTorch)
  2. Group by base_quadrat_id (seasonal aggregation)
  3. Count species frequency across ALL tiles from ALL seasonal images
  4. Apply geographic mask (filter non-allowed species) AFTER aggregation
  5. Take top-K species for submission
  6. Map class indices to species IDs
  7. Write CSV
"""

import os
import re
import csv
import glob
import json
import argparse
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def extract_base_quadrat_id(quadrat_id: str) -> str:
    """
    Strip date suffixes to group seasonal images of the same location.
    Copied from reference: classification/aggregation.py:22-33
    """
    patterns = [
        (r"^(CBN-.*?-.*?)-\d{8}$", 1),
        (r"^(GUARDEN-.*?-.*?)-.*$", 1),
        (r"^(LISAH-.*?)-\d{8}$", 1),
        (r"^(OPTMix-\d+)-.*$", 1),
        (r"^(RNNB-\d+-\d+)-\d{8}$", 1),
    ]
    for pattern, group_idx in patterns:
        match = re.match(pattern, quadrat_id)
        if match:
            return match.group(group_idx)
    return quadrat_id


def load_species_map(path: Optional[str], num_classes: int) -> List[int]:
    if path is None:
        return list(range(num_classes))
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list) or len(arr) != num_classes:
            raise ValueError(f"JSON species map must be a list of length {num_classes}")
        return [int(x) for x in arr]

    df = pd.read_csv(path)
    if "class_index" in df.columns and "species_id" in df.columns:
        c2s = dict(
            zip(df["class_index"].astype(int).tolist(), df["species_id"].astype(int).tolist())
        )
        missing = [i for i in range(num_classes) if i not in c2s]
        if missing:
            raise ValueError(f"Species map missing {len(missing)} indices, first {missing[:10]}")
        return [int(c2s[i]) for i in range(num_classes)]

    raise ValueError("Species map must be json list or csv with class_index and species_id columns")


def load_class_mask(path: str, num_classes: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = np.load(path)
    if m.ndim != 1 or m.shape[0] != num_classes:
        raise ValueError(f"Mask shape {m.shape}, expected ({num_classes},)")
    if m.dtype != np.bool_:
        m = m.astype(np.bool_)
    if not m.any():
        raise ValueError("Class mask has zero allowed classes")
    return m


def format_species_list(species_ids: List[int]) -> str:
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Majority vote + submission from per-tile species (reference paper pipeline)"
    )

    ap.add_argument("--in-parquet-root", required=True,
                    help="Folder with shard_* from extract_testdata_tile_species.py")
    ap.add_argument("--pattern", default="shard_*/*.parquet",
                    help="Glob under in-parquet-root for parquet files")
    ap.add_argument("--out-csv", required=True,
                    help="Output submission CSV path")

    ap.add_argument("--id-col", default="quadrat_id")
    ap.add_argument("--species-col", default="tile_species_json")

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--species-map", default=None,
                    help="JSON or CSV mapping class indices to species IDs")
    ap.add_argument("--class-mask-npy", default=None,
                    help="Geographic mask .npy applied AFTER aggregation")

    ap.add_argument("--topk", type=int, default=10,
                    help="Top-K species per submission row (reference uses 10)")
    ap.add_argument("--min-k", type=int, default=1,
                    help="Minimum species per row")

    args = ap.parse_args()

    # --- Load per-tile species from parquet ---
    files = sorted(glob.glob(os.path.join(args.in_parquet_root, args.pattern)))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found: {args.in_parquet_root}/{args.pattern}"
        )
    print(f"Found {len(files)} parquet files")

    image_tile_species: Dict[str, List[List[int]]] = {}
    for fp in files:
        df = pq.read_table(fp, columns=[args.id_col, args.species_col]).to_pandas()
        for _, r in df.iterrows():
            qid = str(r[args.id_col])
            tile_species = json.loads(r[args.species_col])
            image_tile_species[qid] = tile_species

    print(f"Loaded tile species for {len(image_tile_species)} images")

    # --- Group by base_quadrat_id (seasonal aggregation) ---
    groups: Dict[str, List[str]] = defaultdict(list)
    for qid in image_tile_species:
        groups[extract_base_quadrat_id(qid)].append(qid)

    n_grouped = sum(1 for g in groups.values() if len(g) > 1)
    print(f"Grouped into {len(groups)} base quadrats ({n_grouped} with multiple seasonal images)")

    # --- Majority vote per group ---
    # Flatten all per-tile species across all tiles and all seasonal images -> Counter -> rank
    group_ranked: Dict[str, List[int]] = {}

    for base_id, qids in groups.items():
        counter: Counter = Counter()
        for qid in qids:
            for tile_species_list in image_tile_species[qid]:
                for sp in tile_species_list:
                    counter[sp] += 1
        group_ranked[base_id] = [sp for sp, _ in counter.most_common()]

    # --- Load species map and geo mask ---
    species_map = load_species_map(args.species_map, args.num_classes)

    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask(args.class_mask_npy, args.num_classes)
        print(f"Geo mask: {class_mask.sum()} allowed out of {args.num_classes}")

    # --- Generate submission rows ---
    rows: List[Tuple[str, str]] = []

    for qid in sorted(image_tile_species.keys()):
        base_id = extract_base_quadrat_id(qid)
        ranked = group_ranked[base_id]

        # Apply geo mask AFTER aggregation (matching reference paper)
        if class_mask is not None:
            ranked = [idx for idx in ranked if class_mask[idx]]

        # Take top-K species
        kept = ranked[: max(int(args.topk), int(args.min_k))]

        # Map class indices to species IDs
        kept_species = [species_map[i] for i in kept]
        rows.append((qid, format_species_list(kept_species)))

    # --- Write CSV ---
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, s in rows:
            w.writerow([qid, s])

    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


'''

## No JPEG — with geo mask, top-10 (matching reference paper exactly)

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/generate_submit_from_tile_species.py \
  --in-parquet-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species \
  --pattern "shard_*/*.parquet" \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260304_tile_species_majority_vote_top10_masked_noJPEG.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --class-mask-npy /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/species_masking/sw_europe_mask_bool.npy \
  --topk 10 \
  --min-k 1


## No JPEG — WITHOUT geo mask (for comparison)

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/generate_submit_from_tile_species.py \
  --in-parquet-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species \
  --pattern "shard_*/*.parquet" \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260304_tile_species_majority_vote_top10_nomask_noJPEG.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --topk 10 \
  --min-k 1

'''
