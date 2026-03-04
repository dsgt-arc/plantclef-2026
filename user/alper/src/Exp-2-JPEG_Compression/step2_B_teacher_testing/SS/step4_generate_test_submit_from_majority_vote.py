#!/usr/bin/env python3
"""
Generate submission CSV from majority vote aggregation output.

Matches the reference paper pipeline order:
  1. Read ranked species from majority vote (no geo filter applied yet)
  2. Apply geographic mask (filter out non-allowed species)
  3. Take top-K from remaining species
  4. Map class indices to species IDs
  5. Write CSV
"""

import os
import csv
import json
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
        class_to_species = dict(
            zip(df["class_index"].astype(int).tolist(), df["species_id"].astype(int).tolist())
        )
        missing = [i for i in range(num_classes) if i not in class_to_species]
        if missing:
            raise ValueError(f"Species map missing {len(missing)} indices, first {missing[:10]}")
        return [int(class_to_species[i]) for i in range(num_classes)]

    raise ValueError("Species map must be json list or csv with class_index and species_id columns")


def load_class_mask_npy(path: str, num_classes: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = np.load(path)
    if m.ndim != 1 or m.shape[0] != num_classes:
        raise ValueError(f"class mask must be 1D length {num_classes}, got {m.shape}")
    if m.dtype != np.bool_:
        m = m.astype(np.bool_)
    if not m.any():
        raise ValueError("class mask has zero allowed classes")
    return m


def format_species_list(species_ids: List[int]) -> str:
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate submission CSV from majority vote aggregation output"
    )

    ap.add_argument("--in-parquet", required=True,
                     help="Parquet from run_step2_majority_vote_on_test_set.py")
    ap.add_argument("--out-csv", required=True)

    ap.add_argument("--id-col", default="quadrat_id")
    ap.add_argument("--ranked-col", default="ranked_class_indices")

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--species-map", default=None)

    ap.add_argument("--topk", type=int, default=10,
                     help="Number of top species to include in submission")
    ap.add_argument("--min-k", type=int, default=1,
                     help="Minimum number of species per image")

    ap.add_argument("--class-mask-npy", default=None,
                     help="Geographic mask applied AFTER aggregation (matching reference paper)")

    args = ap.parse_args()

    t = pq.read_table(args.in_parquet, columns=[args.id_col, args.ranked_col]).to_pandas()
    species_map = load_species_map(args.species_map, args.num_classes)

    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)
        print(f"Loaded class mask: {class_mask.sum()} allowed out of {class_mask.shape[0]}")

    rows: List[Tuple[str, str]] = []

    for _, r in t.iterrows():
        qid = str(r[args.id_col])
        ranked_indices = json.loads(r[args.ranked_col])

        # Apply geo mask AFTER aggregation: filter out non-allowed species
        if class_mask is not None:
            ranked_indices = [idx for idx in ranked_indices if class_mask[idx]]

        # Take top-K species
        kept = ranked_indices[: max(int(args.topk), int(args.min_k))]

        # Map class indices to species IDs
        kept_species = [species_map[i] for i in kept]
        rows.append((qid, format_species_list(kept_species)))

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

##  Generate submission CSV (geo mask applied HERE, after aggregation)

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/step4_generate_test_submit_from_majority_vote.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_majority_vote/majority_vote_ranked.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260303_teacher_majority_vote_top10_masked.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --class-mask-npy /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/species_masking/sw_europe_mask_bool.npy \
  --topk 10 \
  --min-k 1

'''
