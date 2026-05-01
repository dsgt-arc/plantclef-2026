"""Audit a PlantCLEF submission CSV for odd / out-of-place species.

Inputs:
- submission CSV (quadrat_id, species_ids list-as-string)
- class_to_species_id.json (Exp-14 pace_downloads)
- species_id_to_name.json
- aquatic_blacklist.json (Exp-14 outputs/signals)
- sw_europe_mask_bool.npy (geo-permitted classes)

Outputs (to stdout):
- frequency table of species predictions (top 30 + bottom edge)
- species predicted that are aquatic / outside SW-Europe geo mask (red flags)
- species predicted only on RNNB / only on alpine sources (cross-source check)
- prediction-count distribution per source prefix
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_PREFIXES = ["RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix", "2024-CEV3"]


def source_of(qid: str) -> str:
    for p in SOURCE_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--class-map",
        default=r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\Exp-14_Adaptive_Temporal_Fusion\pace_downloads\class_to_species_id.json",
    )
    ap.add_argument(
        "--species-name",
        default=r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\Exp-14_Adaptive_Temporal_Fusion\pace_downloads\species_id_to_name.json",
    )
    ap.add_argument(
        "--aquatic-list",
        default=r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\Exp-14_Adaptive_Temporal_Fusion\outputs\signals\aquatic_blacklist.json",
    )
    ap.add_argument(
        "--geo-mask",
        default=r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\Exp-14_Adaptive_Temporal_Fusion\pace_downloads\sw_europe_mask_bool.npy",
    )
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    with open(args.species_name, "r", encoding="utf-8") as f:
        sid2name_raw = json.load(f)
    sid2name = {int(k): v for k, v in sid2name_raw.items()}

    with open(args.class_map, "r", encoding="utf-8") as f:
        c2s_raw = json.load(f)
    if isinstance(c2s_raw, list):
        class_to_sid = {i: int(s) for i, s in enumerate(c2s_raw)}
    else:
        class_to_sid = {int(k): int(v) for k, v in c2s_raw.items()}
    sid_to_class = {v: k for k, v in class_to_sid.items()}

    with open(args.aquatic_list, "r", encoding="utf-8") as f:
        aquatic_raw = json.load(f)
    aquatic_sids: set[int] = set()
    if isinstance(aquatic_raw, list):
        for item in aquatic_raw:
            if isinstance(item, dict):
                if "species_id" in item:
                    aquatic_sids.add(int(item["species_id"]))
            else:
                aquatic_sids.add(int(item))
    elif isinstance(aquatic_raw, dict):
        if "species_ids" in aquatic_raw:
            aquatic_sids.update(int(x) for x in aquatic_raw["species_ids"])
        else:
            for v in aquatic_raw.values():
                if isinstance(v, (list, tuple)):
                    aquatic_sids.update(int(x) for x in v)

    geo_bool = np.load(args.geo_mask).astype(bool)

    df["src"] = df["quadrat_id"].apply(source_of)
    df["preds"] = df["species_ids"].apply(lambda s: ast.literal_eval(s))
    df["n_preds"] = df["preds"].apply(len)

    print("=" * 88)
    print(f"Submission audit: {args.csv}")
    print(f"Rows: {len(df)} quadrats")
    print("=" * 88)

    print("\nPrediction count per source:")
    g = (
        df.groupby("src")
        .agg(
            n_quad=("quadrat_id", "size"),
            mean_n=("n_preds", "mean"),
            median_n=("n_preds", "median"),
            max_n=("n_preds", "max"),
            empty=("n_preds", lambda s: (s == 0).sum()),
        )
        .round(2)
    )
    print(g)

    flat: list[int] = []
    per_src: dict[str, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        flat.extend(row.preds)
        per_src[row.src].extend(row.preds)
    counter = Counter(flat)
    print(f"\nDistinct species predicted: {len(counter)}")
    print(f"Total prediction slots filled: {sum(counter.values())}")

    print(f"\nTop {args.top} most frequent species:")
    print(f"{'count':>6} {'sid':>10}  {'name'}")
    for sid, n in counter.most_common(args.top):
        nm = sid2name.get(sid, "<unknown>")
        flags = []
        if sid in aquatic_sids:
            flags.append("AQUATIC")
        cls = sid_to_class.get(sid)
        if cls is not None and cls < len(geo_bool) and not geo_bool[cls]:
            flags.append("GEO-OUT")
        flag_str = " [" + ",".join(flags) + "]" if flags else ""
        print(f"{n:>6} {sid:>10}  {nm}{flag_str}")

    aquatic_in_sub = [(sid, n) for sid, n in counter.items() if sid in aquatic_sids]
    aquatic_in_sub.sort(key=lambda t: -t[1])
    print(f"\nAquatic-blacklist species appearing in submission: {len(aquatic_in_sub)}")
    for sid, n in aquatic_in_sub[:30]:
        print(f"  {n:>4}x  {sid}  {sid2name.get(sid, '<unknown>')}")

    geo_out = []
    for sid, n in counter.items():
        cls = sid_to_class.get(sid)
        if cls is None:
            continue
        if cls < len(geo_bool) and not geo_bool[cls]:
            geo_out.append((sid, n))
    geo_out.sort(key=lambda t: -t[1])
    print(f"\nGeo-mask-excluded species appearing in submission: {len(geo_out)}")
    for sid, n in geo_out[:30]:
        print(f"  {n:>4}x  {sid}  {sid2name.get(sid, '<unknown>')}")

    sources_for_sid: dict[int, set[str]] = defaultdict(set)
    counts_for_sid_src: dict[tuple[int, str], int] = defaultdict(int)
    for src, lst in per_src.items():
        for sid in lst:
            sources_for_sid[sid].add(src)
            counts_for_sid_src[(sid, src)] += 1

    print("\nSpecies appearing only on RNNB (saltmarsh / coastal-dune) — sanity check:")
    rnnb_only = sorted(
        ((sid, counts_for_sid_src[(sid, "RNNB")]) for sid in counter if sources_for_sid[sid] == {"RNNB"}),
        key=lambda t: -t[1],
    )
    for sid, n in rnnb_only[:25]:
        print(f"  {n:>4}x  {sid}  {sid2name.get(sid, '<unknown>')}")

    print("\nSpecies appearing only on alpine CBN-PdlC — sanity check:")
    cbn_pdlc_only = sorted(
        (
            (sid, counts_for_sid_src[(sid, "CBN-PdlC")])
            for sid in counter
            if sources_for_sid[sid] == {"CBN-PdlC"}
        ),
        key=lambda t: -t[1],
    )
    for sid, n in cbn_pdlc_only[:25]:
        print(f"  {n:>4}x  {sid}  {sid2name.get(sid, '<unknown>')}")

    print("\nSpecies appearing on >= 4 different sources (likely common / weed; sanity check):")
    spread = sorted(
        ((sid, len(sources_for_sid[sid]), counter[sid]) for sid in counter if len(sources_for_sid[sid]) >= 4),
        key=lambda t: (-t[1], -t[2]),
    )
    for sid, k, n in spread[:25]:
        srcs = ",".join(sorted(sources_for_sid[sid]))
        print(f"  {n:>4}x  ({k} src: {srcs})  {sid}  {sid2name.get(sid, '<unknown>')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
