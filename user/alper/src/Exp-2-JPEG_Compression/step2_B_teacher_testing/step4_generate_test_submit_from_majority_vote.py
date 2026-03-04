#!/usr/bin/env python3

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

    args = ap.parse_args()

    t = pq.read_table(args.in_parquet, columns=[args.id_col, args.ranked_col]).to_pandas()
    species_map = load_species_map(args.species_map, args.num_classes)

    rows: List[Tuple[str, str]] = []

    for _, r in t.iterrows():
        qid = str(r[args.id_col])
        ranked_indices = json.loads(r[args.ranked_col])

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
## Step 1: Run majority vote aggregation (with geo mask applied during voting)

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/run_step2_majority_vote_on_test_set.py \
  --in-parquet-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/feature_bank_JPEG_compressed \
  --pattern "shard_*/*.parquet" \
  --out-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_majority_vote/majority_vote_ranked.parquet \
  --logits-col teacher_logits_f16 \
  --num-classes 7806 \
  --n-tiles 16 \
  --per-tile-topk 10 \
  --class-mask-npy /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/species_masking/sw_europe_mask_bool.npy


## Step 2: Generate submission CSV

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/step4_generate_test_submit_from_majority_vote.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_majority_vote/majority_vote_ranked.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260303_teacher_majority_vote_top10_masked_JPEG.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --topk 10 \
  --min-k 1

'''
