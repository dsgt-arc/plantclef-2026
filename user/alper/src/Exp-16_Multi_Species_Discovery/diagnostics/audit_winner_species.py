"""Audit Stage 13 + N3_r0.5 WINNER predictions: do per-source species
make ecological sense?

For each source (alpine, GUARDEN, LISAH, OPTMix, RNNB, etc.):
  - Top-30 most-predicted species (with name)
  - Top-30 species by per-image prediction frequency
  - Genus-level summary (top-15 most-predicted genera)
  - Cross-source cosine of top-N species distributions (sources should
    differ from each other if predictions track real biogeography)
  - Identify obvious oddities: woody species on coastal dune sources,
    aquatic species on alpine, etc.
"""
from __future__ import annotations

import ast
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

WINNER = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
          r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
          r"/outputs/n_floor_sweep/N3_r0.5_WINNER.csv")
SPECIES_NAME = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
                r"/plantclef-2026/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
                r"/pace_downloads/species_id_to_name.json")
OUT_DIR = (r"C:/Users/alper/Documents/PlantCLEF_2026_Local/_GitHub"
           r"/plantclef-2026/user/alper/src/Exp-16_Multi_Species_Discovery"
           r"/outputs/winner_audit")

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
    return name.split()[0] if name else ""


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(SPECIES_NAME, "r", encoding="utf-8") as f:
        sid_to_name = json.load(f)

    df = pd.read_csv(WINNER)
    df["source"] = df["quadrat_id"].apply(source_of)
    df["sids"] = df["species_ids"].apply(ast.literal_eval)
    df["nspp"] = df["sids"].apply(len)

    # Per-source counts
    print("=" * 80)
    print("Per-source basic stats (Winner: F1 0.46168)")
    print("=" * 80)
    print(f"  total quadrats: {len(df)}")
    for src, g in df.groupby("source"):
        ns = g["nspp"]
        print(f"  {src:<12} n={len(g):>4}  mean_pred={ns.mean():.2f}  "
              f"median={ns.median():.0f}  min={ns.min()}  max={ns.max()}")

    # Per-source top-30 species by quadrat-frequency
    print()
    print("=" * 80)
    print("Per-source top-20 most-frequently predicted species")
    print("=" * 80)

    per_src_top: dict = {}
    for src, g in df.groupby("source"):
        c = Counter()
        for sids in g["sids"]:
            for sid in sids:
                c[str(sid)] += 1
        per_src_top[src] = c
        n = len(g)
        print(f"\n  --- {src} (n={n}) ---")
        for sid, cnt in c.most_common(20):
            name = sid_to_name.get(sid, "??")
            pct = 100.0 * cnt / n
            print(f"    [{cnt:>4} = {pct:5.1f}%] {sid}  {name}")

    # Genus-level summary
    print()
    print("=" * 80)
    print("Per-source top-15 genera (by quadrat frequency)")
    print("=" * 80)
    for src, c in per_src_top.items():
        gen = Counter()
        for sid, cnt in c.items():
            name = sid_to_name.get(sid, "")
            g_ = genus_of(name)
            if g_:
                gen[g_] += cnt
        n = sum(1 for q in df["source"] if q == src)
        print(f"\n  --- {src} (n={n}) ---")
        for g_, cnt in gen.most_common(15):
            print(f"    {g_:<25} {cnt:>5}  ({100*cnt/n:5.1f}% of quadrats)")

    # Cross-source distinctiveness: for each pair, top-50 species overlap
    print()
    print("=" * 80)
    print("Cross-source top-50 species overlap (Jaccard)")
    print("(low = sources predict different species, high = sources blur together)")
    print("=" * 80)
    src_top50 = {}
    for src, c in per_src_top.items():
        src_top50[src] = set(s for s, _ in c.most_common(50))
    sources_sorted = sorted(per_src_top.keys())
    print(f"\n    {'':<12} " + " ".join(f"{s:>10}" for s in sources_sorted))
    for s1 in sources_sorted:
        cells = []
        for s2 in sources_sorted:
            if s1 == s2:
                cells.append("    -    ")
                continue
            inter = len(src_top50[s1] & src_top50[s2])
            union = len(src_top50[s1] | src_top50[s2])
            j = inter / max(union, 1)
            cells.append(f"  {j:>6.3f}  ")
        print(f"    {s1:<12} " + " ".join(c for c in cells))

    # Look for ecological oddities
    # Lists of suspicious species per source — substrate-driven hypotheses
    odd_woody = ("Quercus", "Pinus", "Cedrus", "Taxus", "Picea", "Abies",
                 "Larix", "Fagus", "Castanea", "Olea", "Cupressus",
                 "Juniperus", "Pseudotsuga", "Eucalyptus")
    odd_water = ("Posidonia", "Cymodocea", "Zostera", "Halophila",
                 "Najas", "Potamogeton", "Lemna")
    odd_tropical = ("Agave", "Yucca", "Opuntia", "Aloe", "Sarracenia")

    print()
    print("=" * 80)
    print("Ecological oddities by source")
    print("=" * 80)
    for src, c in per_src_top.items():
        n = sum(1 for q in df["source"] if q == src)
        woody_hits = []
        water_hits = []
        trop_hits = []
        for sid, cnt in c.most_common(200):
            name = sid_to_name.get(sid, "")
            for genus in odd_woody:
                if name.startswith(genus + " "):
                    woody_hits.append((sid, name, cnt))
                    break
            for genus in odd_water:
                if name.startswith(genus + " "):
                    water_hits.append((sid, name, cnt))
                    break
            for genus in odd_tropical:
                if name.startswith(genus + " "):
                    trop_hits.append((sid, name, cnt))
                    break
        if woody_hits or water_hits or trop_hits:
            print(f"\n  --- {src} (n={n}) ---")
            if woody_hits:
                print(f"    woody (likely-bg confusion):")
                for sid, name, cnt in woody_hits[:10]:
                    print(f"      [{cnt:>3} = {100*cnt/n:4.1f}%] {name}")
            if water_hits:
                print(f"    aquatic-or-near-water (substrate or geography flag):")
                for sid, name, cnt in water_hits[:10]:
                    print(f"      [{cnt:>3} = {100*cnt/n:4.1f}%] {name}")
            if trop_hits:
                print(f"    tropical/exotic (suspicious):")
                for sid, name, cnt in trop_hits[:10]:
                    print(f"      [{cnt:>3} = {100*cnt/n:4.1f}%] {name}")

    # Save raw counts to JSON
    out = {
        "per_source": {
            src: {sid: cnt for sid, cnt in c.most_common(200)}
            for src, c in per_src_top.items()
        }
    }
    out_path = os.path.join(OUT_DIR, "per_source_top_species.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
