"""Average post-blend ensemble probabilities across multiple variants
and re-emit the threshold + topK CSV.

Each input parquet has [image_name, image_probs_f16] (f16 7806-d
ensemble probabilities, post-blend, post-temporal-fusion). We decode
to f32, average, then apply threshold and topK as in
Exp-11/ensemble_with_knn.generate_csv.

Optional --weights gives per-parquet weights (default uniform).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

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


def load_probs(path: str) -> Dict[str, np.ndarray]:
    df = pq.read_table(path, columns=["image_name", "image_probs_f16"]).to_pandas()
    out: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        out[qid] = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--weights", default=None, help="Comma-separated; default uniform.")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    n = len(args.parquets)
    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")], dtype=np.float32)
    else:
        w = np.ones(n, dtype=np.float32)
    w = w / w.sum()
    print(f"  averaging {n} parquets with weights {w.tolist()}")

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES)

    all_probs: List[Dict[str, np.ndarray]] = []
    for p in args.parquets:
        d = load_probs(p)
        print(f"    {os.path.basename(p)}: {len(d)} quadrats")
        all_probs.append(d)

    common = set(all_probs[0].keys())
    for d in all_probs[1:]:
        common &= set(d.keys())
    print(f"  common: {len(common)} quadrats")

    avg: Dict[str, np.ndarray] = {}
    for qid in common:
        acc = np.zeros(NUM_CLASSES, dtype=np.float32)
        for i, d in enumerate(all_probs):
            acc += w[i] * d[qid]
        avg[qid] = acc.astype(np.float16)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    generate_csv(avg, species_map, class_mask, args.topk, args.threshold, args.out_csv)
    print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
