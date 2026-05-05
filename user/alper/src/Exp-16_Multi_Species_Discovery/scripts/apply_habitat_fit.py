"""Exp-16 — Habitat-fit demotion based on training-data geographic profile.

Mechanism in one sentence: multiply each test prediction's per-class
probability by a habitat-fit factor in [β, 1] derived from how
geographically and altitudinally compatible the species' TRAINING
data distribution is with the test image's source's known regional
locale, then apply N3_r0.5 admission.

This is structurally different from genus-dedup and single-visit-prior
because it imposes EXTERNAL information (the species' training
geographic distribution from PlantCLEF2024 metadata) onto the test
predictions. Cosmopolitan species (observed widely) are not demoted;
range-restricted species observed only in distant regions are.

β = 1.0 → identity (no demotion). Lower β = stronger pull toward
geographically-plausible species.

Audit motivation (winner audit, 2026-04-30 + GenusDedup negative):
- Per-source predictions are ecologically coherent (alpine → alpine
  flora; coastal → halophytes; etc.). Cross-source Jaccard correctly
  clusters sources biogeographically.
- But within-source, suspect predictions exist (OPTMix predicting
  Quercus petraea / Pinus strobus / Plantago subulata).
- Genus-dedup failed because the prediction shape was already near
  optimal at the multi-label-F1 metric. Habitat-fit operates on a
  different axis: instead of cleaning the shape, it tilts the
  underlying probabilities toward geographically plausible species
  before admission. The reshape happens THROUGH the admission rule,
  not after.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
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

SOURCE_REF = {
    "CBN-PdlC":   {"alt_lo": 1500, "alt_hi": 3300},
    "CBN-Pla":    {"alt_lo": 1500, "alt_hi": 3300},
    "CBN-can":    {"alt_lo": 900,  "alt_hi": 2500},
    "GUARDEN":    {"alt_lo": 0,    "alt_hi": 200},
    "RNNB":       {"alt_lo": 0,    "alt_hi": 200},
    "LISAH":      {"alt_lo": 0,    "alt_hi": 400},
    "OPTMix":     {"alt_lo": 50,   "alt_hi": 800},
    "2024-CEV3":  {"alt_lo": 1500, "alt_hi": 3000},
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


def build_habitat_fit_table(
    geo_profile: pd.DataFrame,
    species_map: Dict[int, str],
    beta: float,
    n_near_min: int,
    use_altitude: bool,
    use_refined: bool = False,
    refined_ref: Dict[str, dict] = None,
    radius_km: int = 0,  # 0 = use default columns; 150 or 500 = use _R150/_R500
    beta_overrides: Dict[str, float] = None,
) -> Dict[str, np.ndarray]:
    """For each source, build a (NUM_CLASSES,) array of fit factors in [β,1].

    fit = geo_score * alt_score where:
      geo_score = β + (1-β) * min(1.0, n_near_source / n_near_min)
      alt_score = 1.0 if species' alt_median in [alt_lo - m, alt_hi + m]
                  else max(β, 1.0 - distance/1000m)
                  m = 0.5 * (alt_hi - alt_lo) tolerance.

    Species without training data (n_obs=0) default to fit=1.0.

    beta_overrides: optional per-source β values. If a source key is present,
    its β is used in place of the global β when building that source's fit
    table. β=1.0 disables demotion entirely for that source.
    """
    sid_to_row: Dict[str, dict] = {
        str(r["species_id"]): r.to_dict() for _, r in geo_profile.iterrows()
    }

    src_iter = (refined_ref.items() if use_refined and refined_ref
                else SOURCE_REF.items())
    if use_refined:
        if radius_km in (150, 200, 300, 500):
            col_suffix = f"_refined_R{radius_km}"
        else:
            col_suffix = "_refined"
    else:
        col_suffix = ""
    fit_by_src: Dict[str, np.ndarray] = {}
    for src, ref in src_iter:
        beta_src = (beta_overrides or {}).get(src, beta)
        fit = np.ones(NUM_CLASSES, dtype=np.float32)
        for cls_idx in range(NUM_CLASSES):
            sid = str(species_map[cls_idx])
            r = sid_to_row.get(sid)
            if r is None or r.get("n_obs", 0) == 0:
                continue  # no info -> 1.0 (neutral)
            n_near = r.get(f"n_near_{src}{col_suffix}", 0)
            geo_score = beta_src + (1.0 - beta_src) * min(1.0, n_near / max(n_near_min, 1))
            if use_altitude and not pd.isna(r.get("alt_median", float("nan"))):
                alt_med = r["alt_median"]
                alt_lo = ref["alt_lo"]
                alt_hi = ref["alt_hi"]
                margin = 0.5 * (alt_hi - alt_lo)
                in_band = (alt_lo - margin) <= alt_med <= (alt_hi + margin)
                if in_band:
                    alt_score = 1.0
                else:
                    if alt_med < (alt_lo - margin):
                        d = (alt_lo - margin) - alt_med
                    else:
                        d = alt_med - (alt_hi + margin)
                    alt_score = max(beta_src, 1.0 - d / 1000.0)
            else:
                alt_score = 1.0
            fit[cls_idx] = geo_score * alt_score
        fit_by_src[src] = fit
    return fit_by_src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--geo-profile", required=True,
                    help="Output of build_species_geo_profile.py")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--beta", type=float, required=True,
                    help="Demotion floor in [0,1]. 1.0 = identity.")
    ap.add_argument("--n-near-min", type=int, default=5,
                    help="Training obs near source for full geo score.")
    ap.add_argument("--use-altitude", action="store_true",
                    help="Apply altitude-band demotion in addition to proximity.")
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-floor", type=int, default=3)
    ap.add_argument("--ratio-floor", type=float, default=0.5)
    ap.add_argument("--count-bound-sources", default=",".join(COUNT_BOUND_SOURCES))
    ap.add_argument("--scope-sources", default="ALL",
                    help="Comma-separated source prefixes to demote, or 'ALL'.")
    ap.add_argument("--refined-ref-json", default=None,
                    help="If provided, use refined source centroids from this "
                         "JSON. Looks up n_near_<source>_refined columns.")
    ap.add_argument("--radius-km", type=int, default=0,
                    help="0 = default 250km columns. 150/500 = use precomputed "
                         "_R150/_R500 columns (only with --refined-ref-json).")
    ap.add_argument("--per-source-json", default=None,
                    help="JSON overrides per source. Keys: source prefix. "
                         "Values: dict with any of {threshold, topk, n_floor, "
                         "ratio_floor, count_bound (bool), beta}.")
    args = ap.parse_args()

    cb_sources = tuple(s.strip() for s in args.count_bound_sources.split(","))
    if args.scope_sources == "ALL":
        scope = None
    else:
        scope = set(s.strip() for s in args.scope_sources.split(","))

    print("=" * 72)
    print("Exp-16: habitat-fit demotion")
    print("=" * 72)
    print(f"  ensemble:          {args.ensemble_parquet}")
    print(f"  beta:              {args.beta}")
    print(f"  n_near_min:        {args.n_near_min}")
    print(f"  use_altitude:      {args.use_altitude}")
    print(f"  scope:             {args.scope_sources}")

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    geo_profile = pd.read_parquet(args.geo_profile)
    print(f"\n  loaded geo profile: {len(geo_profile)} species")

    per_source_overrides = None
    if args.per_source_json:
        with open(args.per_source_json, "r", encoding="utf-8") as f:
            per_source_overrides = json.load(f)
        print(f"\n  per-source overrides:")
        for s, cfg in per_source_overrides.items():
            print(f"    {s:<12} {cfg}")

    refined_ref = None
    use_refined = False
    if args.refined_ref_json:
        with open(args.refined_ref_json, "r", encoding="utf-8") as f:
            refined_ref = json.load(f)
        use_refined = True
        print(f"\n  using refined centroids from {args.refined_ref_json}")
        for s, c in refined_ref.items():
            print(f"    {s:<12} lat={c['lat']:.2f}  lon={c['lon']:.2f}  "
                  f"alt=[{c.get('alt_lo','?')},{c.get('alt_hi','?')}]")

    beta_overrides: Dict[str, float] = {}
    if per_source_overrides:
        for s, cfg in per_source_overrides.items():
            if "beta" in cfg:
                beta_overrides[s] = float(cfg["beta"])
    if beta_overrides:
        print(f"\n  per-source β overrides:")
        for s, b in beta_overrides.items():
            print(f"    {s:<12} β={b}")

    fit_by_src = build_habitat_fit_table(
        geo_profile, species_map,
        beta=args.beta, n_near_min=args.n_near_min,
        use_altitude=args.use_altitude,
        use_refined=use_refined, refined_ref=refined_ref,
        radius_km=args.radius_km,
        beta_overrides=beta_overrides,
    )
    # Quick stats on fit distribution
    print(f"\n  habitat-fit table summary:")
    for src in ("CBN-PdlC", "CBN-Pla", "CBN-can", "GUARDEN", "LISAH",
                "OPTMix", "RNNB"):
        if src not in fit_by_src:
            continue
        f = fit_by_src[src]
        n_full = (f >= 0.999).sum()
        n_demoted = (f < 0.999).sum()
        n_floor = (f <= args.beta + 1e-6).sum()
        print(f"    {src:<10} n_full={n_full:>5}  n_demoted={n_demoted:>5}  "
              f"n_at_floor={n_floor:>5}  median_fit={np.median(f):.3f}")

    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    print(f"\n  loaded {len(df)} ensemble rows")

    rows = []
    n_demoted_qids = 0
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        src = source_of(qid)
        p = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES).astype(np.float32)

        if src in ALPINE_SOURCES:
            p[~combined_alpine] = 0.0
        else:
            p[~combined_global] = 0.0

        # Apply habitat-fit demotion
        in_scope = (scope is None) or (src in scope)
        eff_beta = beta_overrides.get(src, args.beta)
        if in_scope and src in fit_by_src and eff_beta < 1.0:
            p = p * fit_by_src[src]
            n_demoted_qids += 1

        # Per-source overrides
        ps = per_source_overrides.get(src, {}) if per_source_overrides else {}
        eff_threshold = ps.get("threshold", args.threshold)
        eff_topk = ps.get("topk", args.topk)
        eff_n_floor = ps.get("n_floor", args.n_floor)
        eff_ratio_floor = ps.get("ratio_floor", args.ratio_floor)
        if "count_bound" in ps:
            is_cb = bool(ps["count_bound"])
        else:
            is_cb = src in cb_sources

        admitted = admit_with_n_floor(
            p, eff_threshold, eff_topk, eff_n_floor, eff_ratio_floor,
            is_cb,
        )
        sids = [species_map[idx] for idx in admitted]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")
    print(f"  rows with habitat-fit applied: {n_demoted_qids}")


if __name__ == "__main__":
    main()
