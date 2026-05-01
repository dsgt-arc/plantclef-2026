"""Exp-16 — Compose post-blend vote-rerank with N-floor admission.

Tests whether the two mechanisms are additive or interfere:
  1. Apply vote-share rerank to the locked Stage 13 ensemble parquet
     (multiplicative reweighting in probability space).
  2. Apply geo + aquatic + per-source biogeo masks.
  3. Apply N-floor admission on count-bound sources (additional
     candidates whose p-ratio to top-1 >= ratio_floor, up to N total).
  4. Emit candidate CSV.

beta=0 + n_floor=0 reproduces Stage 13 winner byte-for-byte.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP14 = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-14_Adaptive_Temporal_Fusion",
))
EXP8 = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-8_Quantile_Aggregation_SS",
))
sys.path.insert(0, EXP14)
sys.path.insert(0, EXP8)

from generate_submission import (  # noqa: E402
    f16_bytes_to_f32_vector, load_class_mask_npy, load_species_map,
)

NUM_CLASSES = 7806
ALPINE_SOURCES = ("CBN-PdlC", "CBN-Pla")
COUNT_BOUND_SOURCES = ("GUARDEN", "LISAH", "OPTMix", "RNNB")
ALL_SRC_PREFIXES = (
    "RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix", "2024-CEV3",
)


def source_of(qid: str) -> str:
    for p in ALL_SRC_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def admit_with_n_floor(
    p: np.ndarray, threshold: float, topk: int,
    n_floor: int, ratio_floor: float, is_count_bound: bool,
) -> List[int]:
    order = np.argsort(p)[::-1]
    standard: List[int] = []
    for idx in order[:topk]:
        if p[idx] >= threshold:
            standard.append(int(idx))
        else:
            break
    if not standard:
        standard = [int(order[0])]
    if not is_count_bound or n_floor <= 0 or len(standard) >= n_floor:
        return standard
    top1_p = float(p[standard[0]])
    if top1_p <= 0.0:
        return standard
    extra: List[int] = []
    target = min(n_floor, topk)
    for idx in order[len(standard):]:
        idx_int = int(idx)
        if len(standard) + len(extra) >= target:
            break
        cand_p = float(p[idx])
        if cand_p <= 0.0 or cand_p / top1_p < ratio_floor:
            break
        if idx_int in standard:
            continue
        extra.append(idx_int)
    return standard + extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--vote-share-parquet", required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--n-floor", type=int, required=True)
    ap.add_argument("--ratio-floor", type=float, required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    print("=" * 72)
    print("Exp-16: vote-rerank + N-floor compose")
    print("=" * 72)
    print(f"  beta:        {args.beta}")
    print(f"  n_floor:     {args.n_floor}")
    print(f"  ratio_floor: {args.ratio_floor}")

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    ens_df = pq.read_table(args.ensemble_parquet,
                           columns=["image_name", "image_probs_f16"]).to_pandas()
    ens_map: Dict[str, np.ndarray] = {}
    for _, r in ens_df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        ens_map[qid] = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES).astype(np.float32)

    vs_df = pd.read_parquet(args.vote_share_parquet,
                            columns=["image_name", "vote_share_f16"])
    vs_map: Dict[str, np.ndarray] = {}
    for _, r in vs_df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        vs_map[qid] = np.frombuffer(
            r["vote_share_f16"], dtype=np.float16).astype(np.float32)

    rows = []
    for qid in sorted(ens_map.keys()):
        p = ens_map[qid].copy()
        if args.beta != 0.0:
            vs = vs_map.get(qid, np.zeros(NUM_CLASSES, dtype=np.float32))
            factor = np.power(vs + args.epsilon, args.beta).astype(np.float32)
            s_orig = float(p.sum())
            p = p * factor
            s_new = float(p.sum())
            if s_new > 0 and s_orig > 0:
                p *= (s_orig / s_new)
            np.clip(p, 0.0, 1.0, out=p)

        src = source_of(qid)
        if src in ALPINE_SOURCES:
            p[~combined_alpine] = 0.0
        else:
            p[~combined_global] = 0.0

        is_cb = src in COUNT_BOUND_SOURCES
        admitted = admit_with_n_floor(
            p, threshold=args.threshold, topk=args.topk,
            n_floor=args.n_floor, ratio_floor=args.ratio_floor,
            is_count_bound=is_cb,
        )
        sids = [species_map[i] for i in admitted]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
