"""Exp-16 — Two-stage habitat-fit (Test 1).

Stage 1: For each test image, classify its habitat from the per-image
top-K predicted species' habitat memberships. The image's habitat is
the one with the highest cumulative species probability among the
top-K predictions.

Stage 2: Apply the habitat-fit demotion using the PREDICTED habitat's
centroid instead of the source's centroid.

Two variants:
  --mode predicted_only   : always use predicted habitat
  --mode source_fallback  : use predicted habitat if confidence
                            (top1_share / top2_share) >= conf_thresh,
                            else fall back to source

β=1.0 still reproduces input parquet (geo_score = 1 everywhere).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

# Source -> habitat mapping (manual, for the source_fallback variant).
# Picked from the dominant_source field of habitat_centroids.json:
#   habitat 0: continental temperate (n=2922, dom 2024-CEV3 0.11)
#   habitat 1: alpine (n=330, dom 2024-CEV3 0.65)
#   habitat 2: med coastal (n=352, dom GUARDEN 0.78)
#   habitat 3: roussillon med (n=182, dom LISAH 0.77)
#   habitat 4: med lowland generalist (n=1100, dom LISAH 0.35)
SOURCE_TO_HABITAT = {
    "CBN-PdlC":  1,  # alpine
    "CBN-Pla":   1,  # alpine
    "CBN-can":   1,  # alpine (sub-alpine ok)
    "GUARDEN":   2,  # med coastal
    "RNNB":      2,  # med coastal
    "LISAH":     4,  # med lowland generalist
    "OPTMix":    0,  # continental temperate
    "2024-CEV3": 1,  # alpine
}


def source_of(qid: str) -> str:
    for p in ALL_SRC_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def admit_with_n_floor(
    p: np.ndarray, threshold: float, topk: int, n_floor: int,
    ratio_floor: float, is_count_bound: bool,
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


def build_fit_table(
    geo_profile: pd.DataFrame,
    species_map: Dict[int, str],
    centroids: Dict[str, dict],
    beta: float,
    n_near_min: int,
    use_altitude: bool,
) -> Dict[int, np.ndarray]:
    """Build (NUM_CLASSES,) fit array for each habitat (key = int)."""
    sid_to_row: Dict[str, dict] = {
        str(r["species_id"]): r.to_dict() for _, r in geo_profile.iterrows()
    }

    fit_by_hab: Dict[int, np.ndarray] = {}
    for k_str, c in centroids.items():
        k = int(k_str)
        fit = np.ones(NUM_CLASSES, dtype=np.float32)
        col = f"n_near_habitat_{k_str}"
        # Altitude band for this habitat
        alt_lo = c.get("alt_lo", c["alt"] - 500.0)
        alt_hi = c.get("alt_hi", c["alt"] + 500.0)
        for cls_idx in range(NUM_CLASSES):
            sid = str(species_map[cls_idx])
            r = sid_to_row.get(sid)
            if r is None or r.get("n_obs", 0) == 0:
                continue
            n_near = int(r.get(col, 0))
            geo_score = beta + (1.0 - beta) * min(1.0, n_near / max(n_near_min, 1))
            if use_altitude and not pd.isna(r.get("alt_median", float("nan"))):
                alt_med = r["alt_median"]
                margin = 0.5 * (alt_hi - alt_lo)
                if (alt_lo - margin) <= alt_med <= (alt_hi + margin):
                    alt_score = 1.0
                else:
                    if alt_med < (alt_lo - margin):
                        d = (alt_lo - margin) - alt_med
                    else:
                        d = alt_med - (alt_hi + margin)
                    alt_score = max(beta, 1.0 - d / 1000.0)
            else:
                alt_score = 1.0
            fit[cls_idx] = geo_score * alt_score
        fit_by_hab[k] = fit
    return fit_by_hab


def build_species_to_habitat(habitat_clusters: pd.DataFrame,
                              species_map: Dict[int, str]) -> np.ndarray:
    """class_idx -> habitat (int), or -1 if species not in any cluster."""
    sid_to_hab: Dict[str, int] = {
        str(r["species_id"]): int(r["habitat"])
        for _, r in habitat_clusters.iterrows()
    }
    out = np.full(NUM_CLASSES, -1, dtype=np.int32)
    for cls_idx in range(NUM_CLASSES):
        sid = str(species_map[cls_idx])
        h = sid_to_hab.get(sid, -1)
        out[cls_idx] = h
    return out


def predict_habitat(p: np.ndarray, cls_to_habitat: np.ndarray, topk: int,
                     n_habitats: int) -> Tuple[int, float, float]:
    """Return (predicted_habitat, top_score, second_score) where the
    score for habitat k is sum(p[idx] for idx in topk if cls_to_habitat[idx]==k).
    """
    order = np.argsort(p)[::-1][:topk]
    h_scores = np.zeros(n_habitats, dtype=np.float32)
    for idx in order:
        h = int(cls_to_habitat[idx])
        if h >= 0:
            h_scores[h] += float(p[idx])
    rank = np.argsort(h_scores)[::-1]
    top = int(rank[0])
    top_s = float(h_scores[top])
    second_s = float(h_scores[rank[1]]) if len(rank) > 1 else 0.0
    return top, top_s, second_s


# Used in type hint above
from typing import Tuple  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--geo-profile", required=True)
    ap.add_argument("--habitat-clusters", required=True)
    ap.add_argument("--habitat-centroids", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mode", choices=["predicted_only", "source_fallback"],
                    required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-near-min", type=int, default=25)
    ap.add_argument("--use-altitude", action="store_true", default=True)
    ap.add_argument("--no-altitude", action="store_false", dest="use_altitude")
    ap.add_argument("--predict-topk", type=int, default=10,
                    help="K for habitat prediction (per-image top-K species).")
    ap.add_argument("--conf-thresh", type=float, default=1.5,
                    help="source_fallback: predicted/second ratio below "
                         "this -> use source-based habitat.")
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-floor", type=int, default=3)
    ap.add_argument("--ratio-floor", type=float, default=0.5)
    ap.add_argument("--count-bound-sources", default=",".join(COUNT_BOUND_SOURCES))
    args = ap.parse_args()

    cb_sources = tuple(s.strip() for s in args.count_bound_sources.split(","))

    print("=" * 72)
    print(f"Exp-16: two-stage habitat-fit ({args.mode})")
    print("=" * 72)
    print(f"  beta:         {args.beta}")
    print(f"  n_near_min:   {args.n_near_min}")
    print(f"  use_altitude: {args.use_altitude}")
    print(f"  predict_topk: {args.predict_topk}")
    if args.mode == "source_fallback":
        print(f"  conf_thresh:  {args.conf_thresh}")

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine

    species_map = load_species_map(args.species_map, NUM_CLASSES)

    print("\nLoading habitat lookups...")
    geo_profile = pd.read_parquet(args.geo_profile)
    habitat_clusters = pd.read_parquet(args.habitat_clusters)
    with open(args.habitat_centroids, "r", encoding="utf-8") as f:
        centroids = json.load(f)
    print(f"  centroids: {len(centroids)} habitats")
    print(f"  species->habitat: {len(habitat_clusters)} species mapped")

    cls_to_habitat = build_species_to_habitat(habitat_clusters, species_map)
    n_classes_mapped = (cls_to_habitat >= 0).sum()
    print(f"  classes mapped to a habitat: {n_classes_mapped} / {NUM_CLASSES}")

    fit_by_hab = build_fit_table(
        geo_profile, species_map, centroids,
        beta=args.beta, n_near_min=args.n_near_min,
        use_altitude=args.use_altitude,
    )

    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    print(f"\n  loaded {len(df)} ensemble rows")

    rows = []
    n_habitats = len(centroids)
    used_predicted = 0
    used_source_fallback = 0
    habitat_dist = {k: 0 for k in range(n_habitats)}
    src_pred_agreement = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        src = source_of(qid)
        p = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES).astype(np.float32)

        if src in ALPINE_SOURCES:
            p[~combined_alpine] = 0.0
        else:
            p[~combined_global] = 0.0

        # Predict habitat from top-K species
        pred_h, top_s, sec_s = predict_habitat(
            p, cls_to_habitat, args.predict_topk, n_habitats,
        )

        # Decide which habitat to use
        if args.mode == "predicted_only":
            chosen_h = pred_h
            used_predicted += 1
        elif args.mode == "source_fallback":
            ratio = top_s / max(sec_s, 1e-6)
            if ratio >= args.conf_thresh:
                chosen_h = pred_h
                used_predicted += 1
            else:
                chosen_h = SOURCE_TO_HABITAT.get(src, pred_h)
                used_source_fallback += 1

        habitat_dist[chosen_h] = habitat_dist.get(chosen_h, 0) + 1
        # Track source vs prediction agreement
        src_h = SOURCE_TO_HABITAT.get(src, -1)
        key = (src, pred_h, src_h == pred_h)
        src_pred_agreement[key] = src_pred_agreement.get(key, 0) + 1

        # Apply demotion using chosen habitat's fit table
        if args.beta < 1.0:
            p = p * fit_by_hab[chosen_h]

        is_cb = src in cb_sources
        admitted = admit_with_n_floor(
            p, args.threshold, args.topk, args.n_floor, args.ratio_floor,
            is_cb,
        )
        sids = [species_map[idx] for idx in admitted]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")
    print(f"  used_predicted: {used_predicted}, "
          f"used_source_fallback: {used_source_fallback}")
    print(f"  predicted-habitat distribution:")
    for k in sorted(habitat_dist.keys()):
        c = centroids[str(k)]
        n_h = habitat_dist[k]
        print(f"    habitat {k} ({c.get('dominant_source','?')}): {n_h}")

    print(f"\n  source vs predicted-habitat agreement summary:")
    by_src = {}
    for (src, pred_h, agreed), c in src_pred_agreement.items():
        if src not in by_src:
            by_src[src] = {"agreed": 0, "disagreed": 0}
        if agreed:
            by_src[src]["agreed"] += c
        else:
            by_src[src]["disagreed"] += c
    for src in sorted(by_src.keys()):
        ag = by_src[src]["agreed"]
        di = by_src[src]["disagreed"]
        tot = ag + di
        if tot:
            print(f"    {src:<12} agreed={ag:>4} / {tot:>4}  "
                  f"({100*ag/tot:.0f}%)")


if __name__ == "__main__":
    main()
