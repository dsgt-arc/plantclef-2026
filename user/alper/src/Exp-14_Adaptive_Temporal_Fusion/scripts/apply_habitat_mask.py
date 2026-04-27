"""Apply a habitat mask to a saved ensemble probability parquet and
re-emit the threshold + topK CSV. Used to add the aquatic-species
blacklist to an existing winner submission post-hoc.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP14 = os.path.dirname(SCRIPT_DIR)
EXP8 = os.path.join(EXP14, "..", "Exp-8_Quantile_Aggregation_SS")
EXP11 = os.path.join(EXP14, "..", "Exp-11_Conv_ViT_Ensemble")
sys.path.insert(0, os.path.abspath(EXP8))
sys.path.insert(0, os.path.abspath(EXP11))

from generate_submission import (  # noqa: E402
    f16_bytes_to_f32_vector, load_class_mask_npy, load_species_map,
)
from ensemble_with_knn import generate_csv  # noqa: E402

NUM_CLASSES = 7806


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    habitat = np.load(args.habitat_mask).astype(np.bool_)
    if habitat.shape[0] != NUM_CLASSES:
        raise ValueError(f"habitat mask shape {habitat.shape} != {NUM_CLASSES}")
    combined = geo & habitat
    print(f"  geo mask passes:      {geo.sum()} / {NUM_CLASSES}")
    print(f"  habitat mask passes:  {habitat.sum()} / {NUM_CLASSES}")
    print(f"  combined mask passes: {combined.sum()} / {NUM_CLASSES}")
    print(f"  newly blocked by habitat: {(geo & ~habitat).sum()} species")

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    probs: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        probs[qid] = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES)
    print(f"  loaded {len(probs)} quadrats")

    generate_csv(probs, species_map, combined, args.topk, args.threshold,
                 args.out_csv)
    print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
