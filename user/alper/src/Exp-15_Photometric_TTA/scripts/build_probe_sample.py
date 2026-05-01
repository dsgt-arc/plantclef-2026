"""Build the Phase-1 probe sample for Exp-15.

Auto-stratified 40-quadrat sample from the Exp-14 density.parquet:
    15 brownish-band       0.05 <= density < 0.15
    10 lush-band           density >= 0.40
    10 very-barren-band    density < 0.05
     5 user-flagged seeds  (RNNB pairs from manual inspection)

Deterministic via numpy.random.RandomState(seed=15). No human review.
Output: outputs/probe/probe_sample.csv with columns
    quadrat_id, density, source, band, flag_reason
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EXP15_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP15_ROOT))
EXP14_ROOT = EXP15_ROOT.parent / "Exp-14_Adaptive_Temporal_Fusion"
sys.path.insert(0, str(EXP14_ROOT))
from signals import parse_source  # noqa: E402  (Exp-14 utility)

USER_SEEDS: List[str] = [
    "RNNB-3-1-20240117",
    "RNNB-3-2-20240117",
    "RNNB-2-7-20240117",
    "RNNB-1-2-20240117",
    "RNNB-3-2-20230512",  # lush sister of one of the above
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--density-parquet",
        default=str(EXP14_ROOT / "outputs" / "signals" / "density.parquet"),
    )
    ap.add_argument(
        "--out",
        default=str(EXP15_ROOT / "outputs" / "probe" / "probe_sample.csv"),
    )
    ap.add_argument("--n-brownish", type=int, default=15)
    ap.add_argument("--n-lush", type=int, default=10)
    ap.add_argument("--n-barren", type=int, default=10)
    ap.add_argument("--seed", type=int, default=15)
    args = ap.parse_args()

    if not os.path.exists(args.density_parquet):
        raise SystemExit(f"density parquet not found: {args.density_parquet}")

    df = pd.read_parquet(args.density_parquet)
    df["source"] = df["quadrat_id"].map(parse_source)

    rng = np.random.RandomState(args.seed)

    def stratified_pick(mask: pd.Series, n: int, label: str) -> pd.DataFrame:
        sub = df[mask].copy()
        if len(sub) == 0:
            return sub.assign(band=label)
        # Stratify by source to avoid all-RNNB or all-CBN samples
        by_src = []
        for src, group in sub.groupby("source"):
            k = max(1, int(round(n * len(group) / len(sub))))
            k = min(k, len(group))
            by_src.append(group.sample(n=k, random_state=rng))
        picked = pd.concat(by_src, ignore_index=True)
        if len(picked) > n:
            picked = picked.sample(n=n, random_state=rng).reset_index(drop=True)
        elif len(picked) < n:
            extra_needed = n - len(picked)
            remaining = sub[~sub["quadrat_id"].isin(picked["quadrat_id"])]
            if len(remaining):
                top_up = remaining.sample(
                    n=min(extra_needed, len(remaining)), random_state=rng
                )
                picked = pd.concat([picked, top_up], ignore_index=True)
        return picked.assign(band=label)

    brownish = stratified_pick(
        (df["density"] >= 0.05) & (df["density"] < 0.15),
        args.n_brownish, "brownish",
    )
    lush = stratified_pick(df["density"] >= 0.40, args.n_lush, "lush")
    barren = stratified_pick(df["density"] < 0.05, args.n_barren, "very_barren")

    seeds = df[df["quadrat_id"].isin(USER_SEEDS)].copy().assign(band="user_seed")

    out = pd.concat([brownish, lush, barren, seeds], ignore_index=True)
    # Drop duplicates (a seed may also fall in a band) — keep first occurrence (band)
    out = out.drop_duplicates(subset=["quadrat_id"], keep="first").reset_index(drop=True)
    out["flag_reason"] = out["band"].map({
        "brownish":    "0.05 <= density < 0.15",
        "lush":        "density >= 0.40",
        "very_barren": "density < 0.05",
        "user_seed":   "manual flag from Exp-14",
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"  wrote {len(out)} quadrats -> {args.out}")
    print("  band distribution:")
    print(out["band"].value_counts().to_string())
    print("  source distribution:")
    print(out["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
