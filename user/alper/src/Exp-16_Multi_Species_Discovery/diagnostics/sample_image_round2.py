"""Round 2 of test image audit — look for properties I haven't checked.

Focuses on:
  - single-visit images (sites with only 1 visit -> no temporal prior)
  - date / month distribution per source
  - image-level color heterogeneity (proxy for content diversity)
  - brightness gradient top-bottom, left-right (uneven lighting indicator)
  - dominant color hue (proxy for season/state)
  - tile-to-tile color homogeneity (within-image variance)

Output: per-source summary + flag specific outlier images.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_DIR = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\images"
TEST_META = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\PlantCLEF2025_test.csv"
OUT_DIR = (r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub"
           r"\plantclef-2026\user\alper\src\Exp-16_Multi_Species_Discovery"
           r"\outputs\image_audit_round2")
SOURCES = ("RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix",
           "2024-CEV3")
N_PER_SOURCE = 30


def source_of(qid: str) -> str:
    for p in SOURCES:
        if qid.startswith(p):
            return p
    return "OTHER"


def site_of(qid: str) -> str:
    parts = qid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return qid


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ---- Test set composition: single-visit detection ----
    print("=" * 80)
    print("Single-visit detection (sites with only 1 photo in test set)")
    print("=" * 80)
    meta = pd.read_csv(TEST_META, sep=";")
    meta["src"] = meta["quadrat_id"].apply(source_of)
    meta["site"] = meta["quadrat_id"].apply(site_of)
    site_counts = meta.groupby("site").size()
    single_sites = set(site_counts[site_counts == 1].index)
    meta["is_single"] = meta["site"].isin(single_sites)
    print(f"  total quadrats:           {len(meta)}")
    print(f"  unique sites:             {meta['site'].nunique()}")
    print(f"  single-visit sites:       {len(single_sites)}")
    print(f"  single-visit quadrats:    {meta['is_single'].sum()}")
    print()
    print(f"  per-source single-visit fraction:")
    for s, g in meta.groupby("src"):
        sv = g["is_single"].sum()
        n = len(g)
        print(f"    {s:<12} {sv:>4} / {n:>4}  ({100*sv/n:.1f}%)")

    # ---- Date / month distribution ----
    print()
    print("=" * 80)
    print("Date distribution per source (months covered)")
    print("=" * 80)
    meta["month"] = pd.to_datetime(meta["date"], errors="coerce").dt.month
    meta["year"] = pd.to_datetime(meta["date"], errors="coerce").dt.year
    print(f"  year range: {meta.year.min():.0f} - {meta.year.max():.0f}")
    print()
    print(f"  per-source month distribution (count by month):")
    print(f"    {'src':<12} " + " ".join(f"{m:>3}" for m in range(1, 13)) + "  total")
    for s, g in meta.groupby("src"):
        counts = [int((g.month == m).sum()) for m in range(1, 13)]
        print(f"    {s:<12} " + " ".join(f"{c:>3}" for c in counts) +
              f"  {len(g):>4}")

    # ---- Visual property audit (round 2) ----
    print()
    print("=" * 80)
    print(f"Round-2 visual properties (sample {N_PER_SOURCE}/source)")
    print("=" * 80)
    rng = random.Random(42)
    files_by_src = defaultdict(list)
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            files_by_src[source_of(f)].append(f)

    results = defaultdict(list)
    for src, files in files_by_src.items():
        n = min(N_PER_SOURCE, len(files))
        sampled = rng.sample(files, n)
        for fn in sampled:
            try:
                img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
                w, h = img.size
                if max(w, h) > 1024:
                    scale = 1024 / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)),
                                     Image.LANCZOS)
                arr = np.asarray(img).astype(np.float32) / 255.0
                gray = arr.mean(axis=2)
                # Brightness gradient: difference top-bottom, left-right
                ht, wd = gray.shape
                top = gray[:ht//4].mean()
                bot = gray[3*ht//4:].mean()
                lft = gray[:, :wd//4].mean()
                rgt = gray[:, 3*wd//4:].mean()
                # Color heterogeneity: std of mean RGB across 4x4 grid blocks
                gh, gw = ht // 4, wd // 4
                block_means = []
                for r in range(4):
                    for c in range(4):
                        blk = arr[r*gh:(r+1)*gh, c*gw:(c+1)*gw].mean(axis=(0, 1))
                        block_means.append(blk)
                block_means = np.array(block_means)
                color_heterogeneity = float(block_means.std(axis=0).mean())
                # Dominant hue (HSV)
                rmax = arr.max(axis=2)
                rmin = arr.min(axis=2)
                delta = rmax - rmin
                hue = np.zeros_like(rmax)
                # Compute hue (only where delta > 0)
                valid = delta > 1e-6
                rd = (rmax - arr[..., 0]) / np.where(valid, delta, 1.0)
                gd = (rmax - arr[..., 1]) / np.where(valid, delta, 1.0)
                bd = (rmax - arr[..., 2]) / np.where(valid, delta, 1.0)
                # Brightness variance
                brightness_var = float(gray.var())
                row = {
                    "image": fn,
                    "vert_grad": top - bot,
                    "horiz_grad": lft - rgt,
                    "color_heterogeneity": color_heterogeneity,
                    "brightness_var": brightness_var,
                    "rgb_b_mean": float(arr[..., 2].mean()),
                    "rgb_g_mean": float(arr[..., 1].mean()),
                    "rgb_r_mean": float(arr[..., 0].mean()),
                    "rg_ratio": float(arr[..., 0].mean() / max(arr[..., 1].mean(), 1e-6)),
                }
                results[src].append(row)
            except Exception as e:
                print(f"  ERR {fn}: {e}")

    # Print summary
    print()
    print(f"  per-source medians (visual features):")
    keys = ("vert_grad", "horiz_grad", "color_heterogeneity",
            "brightness_var", "rg_ratio")
    print(f"    {'source':<12} " + " ".join(f"{k:>20s}" for k in keys))
    summary = {}
    for src in SOURCES:
        if src not in results or not results[src]:
            continue
        rows = results[src]
        med = {k: float(np.median([r[k] for r in rows])) for k in keys}
        std = {k: float(np.std([r[k] for r in rows])) for k in keys}
        print(f"    {src:<12} " +
              " ".join(f"{med[k]:>20.4f}" for k in keys))
        summary[src] = {
            "n": len(rows),
            "median": med,
            "std": std,
            "samples": rows[:5],  # first 5 for visual reference
        }

    # Identify outliers per source
    print()
    print(f"  per-source outliers (|vert_grad| > 0.10):")
    for src, rows in results.items():
        outliers = [r for r in rows if abs(r["vert_grad"]) > 0.10]
        if outliers:
            print(f"    {src}:")
            for r in outliers[:5]:
                print(f"      {r['image']}: vert_grad={r['vert_grad']:+.3f}  "
                      f"horiz_grad={r['horiz_grad']:+.3f}")

    # Cross-source single-visit + date analysis
    print()
    print("=" * 80)
    print("Cross-tabulation: single-visit by month")
    print("=" * 80)
    sv = meta[meta["is_single"]]
    print(f"  total single-visit quadrats: {len(sv)}")
    for s, g in sv.groupby("src"):
        if len(g) > 0:
            month_dist = g["month"].value_counts().sort_index()
            ms = ", ".join(f"M{m}:{c}" for m, c in month_dist.items())
            print(f"    {s}: {ms}")

    out_path = os.path.join(OUT_DIR, "round2_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "single_visit": {
                "total": int(meta["is_single"].sum()),
                "per_source": {s: int(g["is_single"].sum())
                               for s, g in meta.groupby("src")},
            },
            "date_distribution": {
                s: {int(m): int(c) for m, c in g["month"].value_counts().sort_index().items()}
                for s, g in meta.groupby("src")
            },
            "visual_properties": summary,
        }, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
