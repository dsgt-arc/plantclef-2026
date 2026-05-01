"""Exp-16 — Per-genus dedup on the locked Stage 13 + N3_r0.5 winner.

Mechanism in one sentence: for each quadrat's prediction list, group
predicted species by genus; keep the top-prob species of each genus,
and drop any sibling species whose probability ratio to the top-of-
genus is below `ratio` (lenient dedup), or always drop siblings if
`ratio = 1.0` (strict dedup).

Audit motivation (winner audit, 2026-04-30):
- Alpine sources predict Salix on 114-181% of quadrats (S. herbacea +
  S. serpillifolia + S. retusa simultaneously).
- RNNB predicts Helichrysum on 55% of quadrats (4 different species).
- The model is genus-confident, species-uncertain on prostrate dwarf
  willows and Mediterranean everlastings.
- If ground truth has 1 species per genus per quadrat, the duplicates
  cost 1 FP each and reduce F1.

ratio = 0.0 → no dedup (parity with input).
ratio = 1.0 → strict (only top-of-genus survives).
ratio ∈ (0,1) → lenient (sibling kept if its prob >= ratio * top-of-genus).

Operates on the post-blend, pre-mask probability parquet so the
dedup uses the same probabilities the admission rule used.
Then re-applies geo + aquatic + biogeo masks and N3_r0.5 admission
ON THE DEDUPED SHORTLIST.

Output schema matches Stage 13 winner (quadrat_id, species_ids).
"""
from __future__ import annotations

import argparse
import json
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
    "RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix",
    "2024-CEV3", "CBN-can",
)


def source_of(qid: str) -> str:
    for p in ALL_SRC_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def genus_of(name: str) -> str:
    if not name:
        return ""
    return name.split()[0]


def admit_with_n_floor_genus_dedup(
    p: np.ndarray,
    threshold: float,
    topk: int,
    n_floor: int,
    ratio_floor: float,
    is_count_bound: bool,
    cls_to_genus: np.ndarray,
    dedup_ratio: float,
) -> List[int]:
    """N-floor admission applied after genus-dedup over a wide candidate
    pool. If dedup_ratio == 0, no dedup (identity). If 1.0, only top-of-
    genus survives. If 0<r<1, sibling survives iff its p >= r * top-of-genus.
    """
    if dedup_ratio <= 0.0:
        # Identity: original N-floor pathway
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

    # Genus-dedup pathway: scan in descending p, but drop classes whose
    # genus already has a higher-prob representative (modulated by ratio).
    order = np.argsort(p)[::-1]
    # Wide candidate pool: take the top 4*topk classes for dedup consideration.
    # 4× because dedup can drop many; we want enough to refill topk after dedup.
    pool_size = max(topk * 4, 40)
    pool = order[:pool_size]

    deduped: List[int] = []
    genus_top_p: Dict[int, float] = {}
    for idx in pool:
        idx_int = int(idx)
        gen = int(cls_to_genus[idx_int])
        if gen < 0:
            # Unknown genus -> always keep
            deduped.append(idx_int)
            continue
        if gen not in genus_top_p:
            genus_top_p[gen] = float(p[idx_int])
            deduped.append(idx_int)
        else:
            top_g = genus_top_p[gen]
            if top_g <= 0:
                continue
            if float(p[idx_int]) >= dedup_ratio * top_g:
                # lenient: passes the ratio threshold, keep
                deduped.append(idx_int)
            # else: drop the sibling

    # Now apply standard N-floor on the deduped pool, ordered by p desc.
    deduped_p = np.array([p[i] for i in deduped], dtype=np.float32)
    deduped_order = np.argsort(deduped_p)[::-1]
    deduped_sorted = [deduped[i] for i in deduped_order]

    standard: List[int] = []
    for idx in deduped_sorted[:topk]:
        if p[idx] >= threshold:
            standard.append(int(idx))
        else:
            break
    if not standard:
        standard = [int(deduped_sorted[0])]
    if not is_count_bound or n_floor <= 0 or len(standard) >= n_floor:
        return standard
    top1_p = float(p[standard[0]])
    if top1_p <= 0.0:
        return standard
    extra: List[int] = []
    target = min(n_floor, topk)
    for idx in deduped_sorted[len(standard):]:
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


def build_class_to_genus(species_map: Dict[int, str],
                          sid_to_name: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build cls_idx -> genus_id, where genus_id is an int. -1 = unknown."""
    genus_to_id: Dict[str, int] = {}
    cls_to_genus = np.full(NUM_CLASSES, -1, dtype=np.int32)
    for cls_idx in range(NUM_CLASSES):
        sid = species_map[cls_idx]
        name = sid_to_name.get(str(sid), "")
        gen = genus_of(name)
        if not gen:
            continue
        if gen not in genus_to_id:
            genus_to_id[gen] = len(genus_to_id)
        cls_to_genus[cls_idx] = genus_to_id[gen]
    return cls_to_genus, genus_to_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--species-name-json", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--dedup-ratio", type=float, required=True,
                    help="0 = off (identity). 1.0 = strict (only top-of-genus). "
                         "0.7-0.85 = lenient.")
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-floor", type=int, default=3)
    ap.add_argument("--ratio-floor", type=float, default=0.5)
    ap.add_argument("--count-bound-sources", default=",".join(COUNT_BOUND_SOURCES))
    ap.add_argument("--scope-sources", default="ALL",
                    help="Comma-separated source prefixes to dedup, or 'ALL' or "
                         "'ALPINE' (CBN-PdlC,CBN-Pla,CBN-can).")
    args = ap.parse_args()

    cb_sources = tuple(s.strip() for s in args.count_bound_sources.split(","))
    if args.scope_sources == "ALL":
        scope = None  # apply to all
    elif args.scope_sources == "ALPINE":
        scope = {"CBN-PdlC", "CBN-Pla", "CBN-can"}
    else:
        scope = set(s.strip() for s in args.scope_sources.split(","))

    print("=" * 72)
    print("Exp-16: per-genus dedup")
    print("=" * 72)
    print(f"  ensemble:          {args.ensemble_parquet}")
    print(f"  dedup_ratio:       {args.dedup_ratio}")
    print(f"  scope_sources:     {args.scope_sources}")
    print(f"  threshold:         {args.threshold}")
    print(f"  topk:              {args.topk}")
    print(f"  n_floor:           {args.n_floor}")
    print(f"  ratio_floor:       {args.ratio_floor}")

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    with open(args.species_name_json, "r", encoding="utf-8") as f:
        sid_to_name = json.load(f)
    cls_to_genus, genus_to_id = build_class_to_genus(species_map, sid_to_name)
    print(f"\n  built genus map: {len(genus_to_id)} unique genera, "
          f"{(cls_to_genus >= 0).sum()} / {NUM_CLASSES} classes mapped")

    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    print(f"  loaded {len(df)} ensemble rows")

    rows = []
    n_dedup_applied = 0
    n_extras_dropped = 0
    per_src_drops: Dict[str, int] = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        src = source_of(qid)
        p = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES).astype(np.float32)

        if src in ALPINE_SOURCES:
            p[~combined_alpine] = 0.0
        else:
            p[~combined_global] = 0.0

        is_cb = src in cb_sources
        in_scope = (scope is None) or (src in scope)
        eff_dedup = args.dedup_ratio if in_scope else 0.0

        # Identity admission first (for reporting)
        admitted_id = admit_with_n_floor_genus_dedup(
            p, args.threshold, args.topk, args.n_floor, args.ratio_floor,
            is_cb, cls_to_genus, dedup_ratio=0.0,
        )
        admitted = admit_with_n_floor_genus_dedup(
            p, args.threshold, args.topk, args.n_floor, args.ratio_floor,
            is_cb, cls_to_genus, dedup_ratio=eff_dedup,
        )

        if eff_dedup > 0 and admitted != admitted_id:
            n_dedup_applied += 1
            n_dropped = len(set(admitted_id) - set(admitted))
            n_extras_dropped += n_dropped
            per_src_drops[src] = per_src_drops.get(src, 0) + n_dropped

        sids = [species_map[idx] for idx in admitted]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")
    print(f"  rows with dedup-changed admission: {n_dedup_applied}")
    print(f"  total predictions removed by dedup: {n_extras_dropped}")
    print(f"  per-source drops: {per_src_drops}")


if __name__ == "__main__":
    main()
