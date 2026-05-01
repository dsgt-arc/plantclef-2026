"""Exp-16 — Per-source N-floor with confidence-tiered admission.

Mechanism in one sentence: on count-bound sources (GUARDEN, LISAH,
OPTMix, RNNB), if fewer than N_floor species pass the standard
threshold=0.100, admit additional candidates up to N_floor — but only
those whose probability ratio to the top-1 admitted species is above
ratio_floor.

N_floor=0 and ratio_floor=infinity gracefully reproduces the existing
Stage 13 winner byte-for-byte.

Inputs:
- ensemble parquet (post-blend, pre-mask): v_ld12RNNBonly_w0.77_ensemble.parquet
- geo mask, aquatic mask, biogeo per-source mask
- count-bound sources (default GUARDEN, LISAH, OPTMix, RNNB)

Outputs:
- candidate CSV with the new admission rule applied
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    p: np.ndarray,
    threshold: float,
    topk: int,
    n_floor: int,
    ratio_floor: float,
    is_count_bound: bool,
) -> List[int]:
    """Return list of admitted class indices, ordered by descending p."""
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
        if cand_p <= 0.0:
            break
        if cand_p / top1_p < ratio_floor:
            break
        if idx_int in standard:
            continue
        extra.append(idx_int)

    return standard + extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-floor", type=int, required=True,
                    help="Min predictions on count-bound sources. 0 = off (recovers baseline).")
    ap.add_argument("--ratio-floor", type=float, required=True,
                    help="Min p_candidate / p_top1 ratio for admission. "
                         "Effectively infinity if very large.")
    ap.add_argument("--count-bound-sources", default=",".join(COUNT_BOUND_SOURCES))
    args = ap.parse_args()

    cb_sources = tuple(s.strip() for s in args.count_bound_sources.split(","))

    print("=" * 72)
    print("Exp-16: per-source N-floor admission")
    print("=" * 72)
    print(f"  ensemble:          {args.ensemble_parquet}")
    print(f"  threshold:         {args.threshold}")
    print(f"  topk:              {args.topk}")
    print(f"  n_floor:           {args.n_floor}")
    print(f"  ratio_floor:       {args.ratio_floor}")
    print(f"  count-bound src:   {cb_sources}")

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine

    print(f"\n  geo passes:           {geo.sum()} / {NUM_CLASSES}")
    print(f"  aquatic passes:       {aquatic.sum()} / {NUM_CLASSES}")
    print(f"  combined global:      {combined_global.sum()} / {NUM_CLASSES}")
    print(f"  combined alpine:      {combined_alpine.sum()} / {NUM_CLASSES}")

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    print(f"\n  loaded {len(df)} ensemble rows")

    rows = []
    n_floor_triggered = 0
    n_extra_admitted = 0
    per_src_extra: Dict[str, int] = {s: 0 for s in cb_sources}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        src = source_of(qid)
        p = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES).astype(np.float32)

        if src in ALPINE_SOURCES:
            p[~combined_alpine] = 0.0
        else:
            p[~combined_global] = 0.0

        is_cb = src in cb_sources
        admitted = admit_with_n_floor(
            p,
            threshold=args.threshold,
            topk=args.topk,
            n_floor=args.n_floor,
            ratio_floor=args.ratio_floor,
            is_count_bound=is_cb,
        )

        # Tracking
        standard_count = sum(1 for idx in admitted
                             if p[idx] >= args.threshold)
        if is_cb and len(admitted) > standard_count:
            n_floor_triggered += 1
            n_extra_admitted += len(admitted) - standard_count
            per_src_extra[src] = per_src_extra.get(src, 0) + (len(admitted) - standard_count)

        sids = [species_map[idx] for idx in admitted]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")
    print(f"  count-bound quadrats with N-floor triggered: {n_floor_triggered}")
    print(f"  total extra admissions: {n_extra_admitted}")
    print(f"  per-source extras: {per_src_extra}")


if __name__ == "__main__":
    main()
