"""Season-weighted visit pairs.

Cyclical month distance, phase-cyclic so May 2023 and May 2024 align
even at delta=12 months. Output: per-quadrat dict {sibling_qid: weight}
keyed by quadrat_id, saved as a JSON file (~1MB).

Run locally — only needs filename dates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

# `from signals import parse_date` etc.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signals import parse_date  # noqa: E402


def parse_location(qid: str) -> str:
    parts = qid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return qid


def cyclical_month_delta(d1: str, d2: str) -> float:
    """Months between two YYYYMMDD strings, mod 12, then folded to [0, 6]."""
    a = datetime.strptime(d1, "%Y%m%d")
    b = datetime.strptime(d2, "%Y%m%d")
    months = abs((a.year - b.year) * 12 + (a.month - b.month))
    cyc = months % 12
    return min(cyc, 12 - cyc)


def season_weight(d1: str, d2: str, sigma: float = 2.0) -> float:
    folded = cyclical_month_delta(d1, d2)
    return math.exp(-(folded ** 2) / (2.0 * sigma * sigma))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quadrat-ids", required=True,
                    help="Text file: one quadrat_id per line.")
    ap.add_argument("--out", required=True, help="JSON output path.")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian sigma in months. Default 2.")
    args = ap.parse_args()

    with open(args.quadrat_ids) as fh:
        qids = [ln.strip() for ln in fh if ln.strip()]

    by_loc: Dict[str, List[str]] = defaultdict(list)
    for q in qids:
        by_loc[parse_location(q)].append(q)

    out: Dict[str, Dict[str, float]] = {}
    n_pairs = 0
    for _loc, members in by_loc.items():
        for q in members:
            d_q = parse_date(q)
            if d_q is None:
                out[q] = {q: 1.0}
                continue
            row: Dict[str, float] = {}
            for s in members:
                d_s = parse_date(s)
                if d_s is None:
                    row[s] = 1.0 if s == q else 0.0
                else:
                    row[s] = season_weight(d_q, d_s, args.sigma)
                    n_pairs += 1
            out[q] = row

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"  wrote {len(out)} qids, {n_pairs} weighted pairs -> {args.out}")


if __name__ == "__main__":
    main()
