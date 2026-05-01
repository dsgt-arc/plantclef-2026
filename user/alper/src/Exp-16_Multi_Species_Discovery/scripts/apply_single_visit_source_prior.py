"""Exp-16 — Single-visit source-prior blend.

Mechanism in one sentence: for quadrats whose site has only one photo
in the test set (no temporal prior), blend the per-image probability
vector with an empirical per-source mean prior at small alpha to
borrow strength from same-source siblings; multi-visit quadrats are
unchanged.

alpha=0.0 reproduces the input ensemble parquet byte-equivalent.

Audit-grounded rationale (round-2 audit, 2026-04-30):
- 509/2105 quadrats are single-visit. Temporal fusion is a no-op for
  them by construction.
- Per-source single-visit fractions: GUARDEN 100%, OPTMix 100%,
  CBN-can 100%, LISAH 84%, RNNB 16%, alpine 0%.
- Alpine sources (CBN-PdlC, CBN-Pla) are 0% single-visit, so this
  mechanism never touches them. That is the structural reason this
  is unlike Stage 1 sim_w (which failed by globally reshuffling
  including alpine).

Pipeline placement: operates on the post-ensemble-blend, pre-mask,
pre-admission parquet. Then re-applies geo + aquatic + biogeo masks
and N3_r0.5 admission to produce the final CSV.

Output schema matches Stage 13 winner CSVs (quadrat_id, species_ids).
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
    "CBN-can",
)


def source_of(qid: str) -> str:
    for p in ALL_SRC_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def site_of(qid: str) -> str:
    parts = qid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return qid


def admit_with_n_floor(
    p: np.ndarray,
    threshold: float,
    topk: int,
    n_floor: int,
    ratio_floor: float,
    is_count_bound: bool,
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
    ap.add_argument("--test-meta", required=True,
                    help="PlantCLEF2025_test.csv (semicolon-separated).")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="Blend weight for source prior. 0 = identity.")
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-floor", type=int, default=3)
    ap.add_argument("--ratio-floor", type=float, default=0.5)
    ap.add_argument("--count-bound-sources", default=",".join(COUNT_BOUND_SOURCES))
    ap.add_argument("--prior-mode", default="all_source",
                    choices=["all_source", "multi_visit_only"],
                    help="all_source: prior is mean over all images of source. "
                         "multi_visit_only: prior is mean over multi-visit images only.")
    args = ap.parse_args()

    cb_sources = tuple(s.strip() for s in args.count_bound_sources.split(","))

    print("=" * 72)
    print("Exp-16: single-visit source-prior blend")
    print("=" * 72)
    print(f"  ensemble:          {args.ensemble_parquet}")
    print(f"  alpha:             {args.alpha}")
    print(f"  prior_mode:        {args.prior_mode}")
    print(f"  threshold:         {args.threshold}")
    print(f"  n_floor:           {args.n_floor}")
    print(f"  ratio_floor:       {args.ratio_floor}")

    # Identify single-visit sites from test metadata
    meta = pd.read_csv(args.test_meta, sep=";")
    meta["site"] = meta["quadrat_id"].apply(site_of)
    site_counts = meta.groupby("site").size()
    single_sites = set(site_counts[site_counts == 1].index)
    qid_to_single: Dict[str, bool] = {}
    for _, r in meta.iterrows():
        qid = str(r["quadrat_id"])
        qid_to_single[qid] = (site_of(qid) in single_sites)
    n_single = sum(qid_to_single.values())
    print(f"\n  single-visit quadrats: {n_single} / {len(meta)}")

    # Load masks
    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine

    species_map = load_species_map(args.species_map, NUM_CLASSES)

    # Load ensemble parquet
    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    print(f"  loaded {len(df)} ensemble rows")

    # Decode all probability vectors and compute per-source means
    # (image probs in this parquet are POST geo-mask already? Let's check
    #  by reading what the full pipeline does. The ensemble parquet stores
    #  P_blend = w*P_vit + (1-w)*P_cnxt before final masks. We compute
    #  source mean on the raw probs to keep prior agnostic; masks applied
    #  AFTER the blend per-row.)
    print("\n  decoding probability matrix...")
    probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    qids = []
    is_single_arr = np.zeros(len(df), dtype=np.bool_)
    src_arr = []
    for i, (_, r) in enumerate(df.iterrows()):
        qid = os.path.splitext(str(r["image_name"]))[0]
        qids.append(qid)
        probs[i] = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES).astype(np.float32)
        is_single_arr[i] = qid_to_single.get(qid, False)
        src_arr.append(source_of(qid))
    src_arr = np.array(src_arr)

    print(f"  decoded {probs.shape}")
    print(f"  is_single rows in ensemble: {is_single_arr.sum()}")

    # Per-source mean prior
    print("\n  per-source mean prior (n samples, top-1 species id):")
    src_priors: Dict[str, np.ndarray] = {}
    for src in sorted(set(src_arr.tolist())):
        if args.prior_mode == "multi_visit_only":
            mask = (src_arr == src) & (~is_single_arr)
            if mask.sum() == 0:
                # Fallback: use all_source if no multi-visit
                mask = (src_arr == src)
        else:
            mask = (src_arr == src)
        if mask.sum() == 0:
            continue
        prior = probs[mask].mean(axis=0)
        # Re-normalize so prior sums to ~same total as a typical row
        # (probs are already per-image probabilities; just store the mean)
        src_priors[src] = prior.astype(np.float32)
        top_idx = int(np.argmax(prior))
        top_sid = species_map[top_idx]
        print(f"    {src:<12} n={mask.sum():>4}  top1=cls{top_idx} (sid={top_sid})  "
              f"top1_p={prior[top_idx]:.4f}")

    # Apply blend per-row, then masks, then N-floor admission
    rows = []
    n_blended = 0
    n_floor_triggered = 0
    n_extra_admitted = 0
    per_src_extra: Dict[str, int] = {s: 0 for s in cb_sources}
    for i, qid in enumerate(qids):
        src = src_arr[i]
        p = probs[i].copy()
        if is_single_arr[i] and src in src_priors and args.alpha > 0.0:
            p = (1.0 - args.alpha) * p + args.alpha * src_priors[src]
            n_blended += 1

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
        standard_count = sum(1 for idx in admitted
                             if p[idx] >= args.threshold)
        if is_cb and len(admitted) > standard_count:
            n_floor_triggered += 1
            n_extra_admitted += len(admitted) - standard_count
            per_src_extra[src] = per_src_extra.get(src, 0) + (
                len(admitted) - standard_count)

        sids = [species_map[idx] for idx in admitted]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")
    print(f"  rows blended (single-visit, alpha>0): {n_blended}")
    print(f"  count-bound quadrats with N-floor triggered: {n_floor_triggered}")
    print(f"  total extra admissions: {n_extra_admitted}")
    print(f"  per-source extras: {per_src_extra}")


if __name__ == "__main__":
    main()
